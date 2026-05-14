import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, is_scipy_available, logging
from .scheduling_utils import SchedulerMixin


if is_scipy_available():
    import scipy.stats

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class CABSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class CABScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
        use_dynamic_shifting (`bool`, defaults to False):
            Whether to apply timestep shifting on-the-fly based on the image resolution.
        base_shift (`float`, defaults to 0.5):
            Value to stabilize image generation. Increasing `base_shift` reduces variation and image is more consistent
            with desired output.
        max_shift (`float`, defaults to 1.15):
            Value change allowed to latent vectors. Increasing `max_shift` encourages more variation and image may be
            more exaggerated or stylized.
        base_image_seq_len (`int`, defaults to 256):
            The base image sequence length.
        max_image_seq_len (`int`, defaults to 4096):
            The maximum image sequence length.
        invert_sigmas (`bool`, defaults to False):
            Whether to invert the sigmas.
        shift_terminal (`float`, defaults to None):
            The end value of the shifted timestep schedule.
        use_karras_sigmas (`bool`, defaults to False):
            Whether to use Karras sigmas for step sizes in the noise schedule during sampling.
        use_exponential_sigmas (`bool`, defaults to False):
            Whether to use exponential sigmas for step sizes in the noise schedule during sampling.
        use_beta_sigmas (`bool`, defaults to False):
            Whether to use beta sigmas for step sizes in the noise schedule during sampling.
        time_shift_type (`str`, defaults to "exponential"):
            The type of dynamic resolution-dependent timestep shifting to apply. Either "exponential" or "linear".
        stochastic_sampling (`bool`, defaults to False):
            Whether to use stochastic sampling.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        base_shift: Optional[float] = 0.5,
        max_shift: Optional[float] = 1.15,
        base_image_seq_len: Optional[int] = 256,
        max_image_seq_len: Optional[int] = 4096,
        invert_sigmas: bool = False,
        shift_terminal: Optional[float] = None,
        use_karras_sigmas: Optional[bool] = False,
        use_exponential_sigmas: Optional[bool] = False,
        use_beta_sigmas: Optional[bool] = False,
        time_shift_type: str = "exponential",
        stochastic_sampling: bool = False,
        solver_order: int = 2,
        theta: float = 0.9,
        prediction_type: str = "flow_prediction",
        algorithm_type: str = "cab",
        use_flow_sigmas: bool = True,
        variance_type: Optional[str] = None,
    ):
        if self.config.use_beta_sigmas and not is_scipy_available():
            raise ImportError("Make sure to install scipy if you want to use beta sigmas.")
        if sum([self.config.use_beta_sigmas, self.config.use_exponential_sigmas, self.config.use_karras_sigmas]) > 1:
            raise ValueError(
                "Only one of `config.use_beta_sigmas`, `config.use_exponential_sigmas`, `config.use_karras_sigmas` can be used."
            )
        if time_shift_type not in {"exponential", "linear"}:
            raise ValueError("`time_shift_type` must either be 'exponential' or 'linear'.")
        self.last_sample = None
        self.last_model = None
        self.eps_bn = 1e-12
        self.prev_eps = None
        self.prev_prev_eps = None
        self.prev_h_lambda = None
        self.prev_prev_h_lambda = None

        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)
        self.order = solver_order
        self.theta = theta


        sigmas = timesteps / num_train_timesteps
        if not use_dynamic_shifting:
            # when use_dynamic_shifting is True, we apply the timestep shifting on the fly based on the image resolution
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps

        self._step_index = None
        self._begin_index = None

        self._shift = shift

        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    @property
    def shift(self):
        """
        The value used for shifting.
        """
        return self._shift

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def set_shift(self, shift: float):
        self._shift = shift

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Forward process in flow-matching

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=sample.device, dtype=sample.dtype)

        if sample.device.type == "mps" and torch.is_floating_point(timestep):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(sample.device, dtype=torch.float32)
            timestep = timestep.to(sample.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(sample.device)
            timestep = timestep.to(sample.device)

        # self.begin_index is None when scheduler is used for training, or pipeline does not implement set_begin_index
        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timestep]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timestep.shape[0]
        else:
            # add noise is called before first denoising step to create initial latent(img2img)
            step_indices = [self.begin_index] * timestep.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)

        sample = sigma * noise + (1.0 - sigma) * sample

        return sample

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps
        
        
    def _sigma_to_alpha_sigma_t(self, sigma: torch.Tensor):
        """
        DPM-style alpha/sigma conversion.
    
        Flow:
            alpha = 1 - sigma
            sigma_t = sigma
    
        Non-flow DPM:
            alpha = 1 / sqrt(1 + sigma^2)
            sigma_t = sigma * alpha
        """
        if self.config.use_flow_sigmas or self.config.prediction_type == "flow_prediction":
            alpha_t = 1.0 - sigma
            sigma_t = sigma
        else:
            alpha_t = 1.0 / torch.sqrt(sigma**2 + 1.0)
            sigma_t = sigma * alpha_t
    
        return alpha_t, sigma_t
    
    
    def _expand_to_sample(self, x, sample):
        while len(x.shape) < len(sample.shape):
            x = x.unsqueeze(-1)
        return x
    
    
    def convert_model_output(self, model_output, sample, sigma):
        """
        DPM-style conversion.
    
        algorithm_type="cab++":
            CAB uses data prediction x0, like DPM++.
    
        algorithm_type="cab":
            CAB uses noise prediction eps, like DPM-Solver.
    
        prediction_type can be:
            flow_prediction, epsilon, sample, v_prediction
        """
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
    
        alpha_t = self._expand_to_sample(alpha_t, sample)
        sigma_t = self._expand_to_sample(sigma_t, sample)
    
        alpha_safe = self._safe_preserve_sign(alpha_t, self.eps_bn)
        sigma_safe = self._safe_preserve_sign(sigma_t, self.eps_bn)
    
        if self.config.variance_type in ["learned", "learned_range"]:
            model_output = model_output[:, : sample.shape[1]]
    
        if self.config.prediction_type == "flow_prediction":
            # flow v = eps - x0
            x0 = sample - sigma_t * model_output
            eps = sample + alpha_t * model_output
    
        elif self.config.prediction_type == "epsilon":
            eps = model_output
            x0 = (sample - sigma_t * eps) / alpha_safe
    
        elif self.config.prediction_type == "sample":
            x0 = model_output
            eps = (sample - alpha_t * x0) / sigma_safe
    
        elif self.config.prediction_type == "v_prediction":
            x0 = alpha_t * sample - sigma_t * model_output
            eps = alpha_t * model_output + sigma_t * sample
    
        else:
            raise ValueError(f"Unknown prediction_type: {self.config.prediction_type}")
    
        if self.config.algorithm_type == "cab++":
            return x0
        elif self.config.algorithm_type == "cab":
            return eps
        else:
            raise ValueError("algorithm_type must be 'cab' or 'cab++'")


    def _safe_preserve_sign(self, x, eps=1e-12):
        return torch.where(
            torch.abs(x) < eps,
            torch.sign(x) * eps + (x == 0).to(x.dtype) * eps,
            x,
        )

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        if self.config.time_shift_type == "exponential":
            return self._time_shift_exponential(mu, sigma, t)
        elif self.config.time_shift_type == "linear":
            return self._time_shift_linear(mu, sigma, t)

    def stretch_shift_to_terminal(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Stretches and shifts the timestep schedule to ensure it terminates at the configured `shift_terminal` config
        value.

        Reference:
        https://github.com/Lightricks/LTX-Video/blob/a01a171f8fe3d99dce2728d60a73fecf4d4238ae/ltx_video/schedulers/rf.py#L51

        Args:
            t (`torch.Tensor`):
                A tensor of timesteps to be stretched and shifted.

        Returns:
            `torch.Tensor`:
                A tensor of adjusted timesteps such that the final value equals `self.config.shift_terminal`.
        """
        one_minus_z = 1 - t
        scale_factor = one_minus_z[-1] / (1 - self.config.shift_terminal)
        stretched_t = 1 - (one_minus_z / scale_factor)
        return stretched_t
        


    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
        timesteps: Optional[List[float]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`, *optional*):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            sigmas (`List[float]`, *optional*):
                Custom values for sigmas to be used for each diffusion step. If `None`, the sigmas are computed
                automatically.
            mu (`float`, *optional*):
                Determines the amount of shifting applied to sigmas when performing resolution-dependent timestep
                shifting.
            timesteps (`List[float]`, *optional*):
                Custom values for timesteps to be used for each diffusion step. If `None`, the timesteps are computed
                automatically.
        """
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError("`mu` must be passed when `use_dynamic_shifting` is set to be `True`")

        if sigmas is not None and timesteps is not None:
            if len(sigmas) != len(timesteps):
                raise ValueError("`sigmas` and `timesteps` should have the same length")

        if num_inference_steps is not None:
            if (sigmas is not None and len(sigmas) != num_inference_steps) or (
                timesteps is not None and len(timesteps) != num_inference_steps
            ):
                raise ValueError(
                    "`sigmas` and `timesteps` should have the same length as num_inference_steps, if `num_inference_steps` is provided"
                )
        else:
            num_inference_steps = len(sigmas) if sigmas is not None else len(timesteps)

        self.num_inference_steps = num_inference_steps

        # 1. Prepare default sigmas
        is_timesteps_provided = timesteps is not None

        if is_timesteps_provided:
            timesteps = np.array(timesteps).astype(np.float32)

        if sigmas is None:
            if timesteps is None:
                timesteps = np.linspace(
                    self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
                )
            sigmas = timesteps / self.config.num_train_timesteps
        else:
            sigmas = np.array(sigmas).astype(np.float32)
            num_inference_steps = len(sigmas)


        if self.config.use_karras_sigmas:
            sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
        elif self.config.use_exponential_sigmas:
            sigmas = self._convert_to_exponential(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
        elif self.config.use_beta_sigmas:
            sigmas = self._convert_to_beta(in_sigmas=sigmas, num_inference_steps=num_inference_steps)

        # 5. Convert sigmas and timesteps to tensors and move to specified device
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        if not is_timesteps_provided:
            timesteps = sigmas * self.config.num_train_timesteps
        else:
            timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32, device=device)

        if timesteps.shape[0] >= 1:
            timesteps = timesteps[:-1]
            sigmas = sigmas[:-1]
        # update step count
        num_inference_steps = timesteps.shape[0]
        self.num_inference_steps = num_inference_steps

        if self.config.invert_sigmas:
            sigmas = 1.0 - sigmas
            timesteps = sigmas * self.config.num_train_timesteps
            sigmas = torch.cat([sigmas, torch.ones(1, device=sigmas.device)])
        else:
            sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
         

        self.timesteps = timesteps
        self.sigmas = sigmas
        self._step_index = None
        self._begin_index = None

        self.last_sample = None
        self.last_model = None
        


        self.prev_eps = None
        self.prev_prev_eps = None
        self.prev_h_lambda = None
        self.prev_prev_h_lambda = None
        



    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()
        
        

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index


    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        theta: float = None,
        generator: Optional[torch.Generator] = None,
        per_token_timesteps: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[CABSchedulerOutput, Tuple]:
    
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError("Pass one of scheduler.timesteps, not integer index.")
    
        if self.step_index is None:
            self._init_step_index(timestep)
        sample_dtype = sample.dtype    
        sample = sample.to(torch.float32)
        model_output = model_output.to(torch.float32)
    
        if theta is None:
            theta = float(getattr(self, "theta", 0.0))
    
        sigma_idx = self.step_index
        t_cur = self.sigmas[sigma_idx].to(sample.device, sample.dtype)
        t_next = self.sigmas[sigma_idx + 1].to(sample.device, sample.dtype)
        dt = t_next - t_cur
    
        alpha_cur, sigma_cur = self._sigma_to_alpha_sigma_t(t_cur)
        alpha_next, sigma_next = self._sigma_to_alpha_sigma_t(t_next)
    
        alpha_cur = self._expand_to_sample(alpha_cur, sample)
        sigma_cur = self._expand_to_sample(sigma_cur, sample)
        alpha_next = self._expand_to_sample(alpha_next, sample)
        sigma_next = self._expand_to_sample(sigma_next, sample)
    
        alpha_cur_safe = self._safe_preserve_sign(alpha_cur, self.eps_bn)
        alpha_next_safe = self._safe_preserve_sign(alpha_next, self.eps_bn)
    
        # YOUR lambda math, not DPM log lambda
        lambda_cur = sigma_cur / alpha_cur_safe
        lambda_next = sigma_next / alpha_next_safe
        h_lambda = lambda_next - lambda_cur
    
        # General converted quantity
        if self.config.prediction_type == "flow_prediction":
            eps_cur = sample + alpha_cur * model_output
        else:
            eps_cur = self.convert_model_output(model_output, sample, t_cur)
    
        y_cur = sample / alpha_cur_safe
        order = int(getattr(self, "order", 2))
    
        # ---------------------------------------------------------
        # Step 0: always Euler in t-space
        # ---------------------------------------------------------
        if self.last_model is None:
            prev_sample = sample + dt * model_output
    
        # ---------------------------------------------------------
        # Step 1:
        # flow      -> AB2 in t-space
        # non-flow  -> non-uniform AB2 in lambda-space
        # ---------------------------------------------------------
        elif self.prev_prev_eps is None:
    
            if self.config.prediction_type == "flow_prediction":
                prev_sample = sample + dt * (
                    1.5 * model_output - 0.5 * self.last_model
                )
            else:
                h_prev_safe = self._safe_preserve_sign(self.prev_h_lambda, self.eps_bn)
                step_ratio = h_lambda / h_prev_safe
    
                y_next = y_cur + h_lambda * (
                    (1.0 + 0.5 * step_ratio) * eps_cur
                    - 0.5 * step_ratio * self.prev_eps
                )
                prev_sample = alpha_next_safe * y_next
    
            # initialize history
            t_prev = self.sigmas[self.step_index - 1].to(sample.device, sample.dtype)
    
            alpha_prev, sigma_prev = self._sigma_to_alpha_sigma_t(t_prev)
            alpha_prev = self._expand_to_sample(alpha_prev, sample)
            sigma_prev = self._expand_to_sample(sigma_prev, sample)
    
            alpha_prev_safe = self._safe_preserve_sign(alpha_prev, self.eps_bn)
            lambda_prev = sigma_prev / alpha_prev_safe
            h_prev_lambda = lambda_cur - lambda_prev
    
            if self.config.prediction_type == "flow_prediction":
                eps_prev = self.last_sample + alpha_prev * self.last_model
            else:
                eps_prev = self.convert_model_output(self.last_model, self.last_sample, t_prev)
    
            self.prev_prev_eps = eps_prev.detach().clone()
            self.prev_eps = eps_cur.detach().clone()
    
            self.prev_prev_h_lambda = h_prev_lambda.detach().clone()
            self.prev_h_lambda = h_lambda.detach().clone()
    
        # ---------------------------------------------------------
        # Step >= 2: CAB in your lambda-space
        # ---------------------------------------------------------
        else:
            h_prev_safe = self._safe_preserve_sign(self.prev_h_lambda, self.eps_bn)
    
            if order == 2:
                step_ratio = h_lambda / h_prev_safe
    
                y_pred = y_cur + h_lambda * (
                    (1.0 + 0.5 * step_ratio) * eps_cur
                    - 0.5 * step_ratio * self.prev_eps
                )
    
            else:
                h_prev_prev_safe = self._safe_preserve_sign(self.prev_prev_h_lambda, self.eps_bn)
                h_sum_safe = self._safe_preserve_sign(
                    self.prev_h_lambda + self.prev_prev_h_lambda,
                    self.eps_bn,
                )
    
                beta0 = (
                    (h_lambda * h_lambda) / 3.0
                    + 0.5 * h_lambda * (2.0 * self.prev_h_lambda + self.prev_prev_h_lambda)
                    + self.prev_h_lambda * (self.prev_h_lambda + self.prev_prev_h_lambda)
                ) / (h_prev_safe * h_sum_safe)
    
                beta1 = -h_lambda * (
                    2.0 * h_lambda
                    + 3.0 * self.prev_h_lambda
                    + 3.0 * self.prev_prev_h_lambda
                ) / (6.0 * h_prev_safe * h_prev_prev_safe)
    
                beta2 = h_lambda * (
                    2.0 * h_lambda + 3.0 * self.prev_h_lambda
                ) / (6.0 * h_prev_prev_safe * h_sum_safe)
    
                y_pred = y_cur + h_lambda * (
                    beta0 * eps_cur
                    + beta1 * self.prev_eps
                    + beta2 * self.prev_prev_eps
                )
    
            h_prev_prev_safe = self._safe_preserve_sign(self.prev_prev_h_lambda, self.eps_bn)
            prev_ratio = self.prev_h_lambda / h_prev_prev_safe
    
            eps_ext = (1.0 + prev_ratio) * self.prev_eps - prev_ratio * self.prev_prev_eps
            defect = eps_cur - eps_ext
    
            y_next = y_pred + theta * h_lambda * defect
            prev_sample = alpha_next_safe * y_next
    
            self.prev_prev_eps = self.prev_eps.detach().clone()
            self.prev_eps = eps_cur.detach().clone()
    
            self.prev_prev_h_lambda = self.prev_h_lambda.detach().clone()
            self.prev_h_lambda = h_lambda.detach().clone()
    
        # Store current model/sample
        self.last_model = model_output.detach().clone()
        self.last_sample = sample.detach().clone()
    
       
        # For flow, keep prev_eps=None so second step becomes t-space AB2.
        if (
            self.config.prediction_type != "flow_prediction"
            and self.prev_eps is None
            and self.last_model is not None
        ):
            self.prev_eps = eps_cur.detach().clone()
            self.prev_h_lambda = h_lambda.detach().clone()
    
        self._step_index += 1
    
        prev_sample = prev_sample.to(sample_dtype)
    
        if not return_dict:
            return (prev_sample,)
    
        return CABSchedulerOutput(prev_sample=prev_sample)

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_karras
    def _convert_to_karras(self, in_sigmas: torch.Tensor, num_inference_steps) -> torch.Tensor:
        """Constructs the noise schedule of Karras et al. (2022)."""

        # Hack to make sure that other schedulers which copy this function don't break
        # TODO: Add this logic to the other schedulers
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.config.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.config.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        rho = 7.0  # 7.0 is the value used in the paper
        ramp = np.linspace(0, 1, num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_exponential
    def _convert_to_exponential(self, in_sigmas: torch.Tensor, num_inference_steps: int) -> torch.Tensor:
        """Constructs an exponential noise schedule."""

        # Hack to make sure that other schedulers which copy this function don't break
        # TODO: Add this logic to the other schedulers
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.config.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.config.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        sigmas = np.exp(np.linspace(math.log(sigma_max), math.log(sigma_min), num_inference_steps))
        return sigmas

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_beta
    def _convert_to_beta(
        self, in_sigmas: torch.Tensor, num_inference_steps: int, alpha: float = 0.6, beta: float = 0.6
    ) -> torch.Tensor:
        """From "Beta Sampling is All You Need" [arXiv:2407.12173] (Lee et. al, 2024)"""

        # Hack to make sure that other schedulers which copy this function don't break
        # TODO: Add this logic to the other schedulers
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.config.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.config.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        sigmas = np.array(
            [
                sigma_min + (ppf * (sigma_max - sigma_min))
                for ppf in [
                    scipy.stats.beta.ppf(timestep, alpha, beta)
                    for timestep in 1 - np.linspace(0, 1, num_inference_steps)
                ]
            ]
        )
        return sigmas

    def _time_shift_exponential(self, mu, sigma, t):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def _time_shift_linear(self, mu, sigma, t):
        return mu / (mu + (1 / t - 1) ** sigma)

    def __len__(self):
        return self.config.num_train_timesteps
