"""Model-owned observation policies for shared and fixed LDS emissions."""

import torch

from ._distributions import matrix_normal_inv_wishart


class ObservationHistory:
    """List-shaped compatibility view of one model-owned observation matrix.

    The LDS code indexes C and Sigma by beat. Shared/fixed policies instead
    expose their current value for every index without storing beat copies.
    """

    def __init__(self, owner, name, emission):
        self.owner = owner
        self.name = name
        self.emission = emission

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [getattr(self.owner, self.name) for _ in range(len(self))[index]]
        return getattr(self.owner, self.name)

    def __setitem__(self, index, value):
        # Cluster-local LDS update paths cannot replace model-owned values.
        pass

    def __len__(self):
        return len(self.emission.A)

    def __iter__(self):
        for _ in range(len(self)):
            yield getattr(self.owner, self.name)

    def append(self, value):
        pass

    def copy(self):
        return [getattr(self.owner, self.name)] * len(self)


class ObservationPolicy:
    """One C/Sigma per output, shared by every cluster in that output."""

    def __init__(self, scope, initial_C, initial_Sigma, degrees_of_freedom):
        self.scope = scope
        self.C = torch.as_tensor(initial_C, dtype=torch.float64).clone()
        self.Sigma = torch.as_tensor(initial_Sigma, dtype=torch.float64).clone()
        precision = torch.eye(self.C.shape[1], dtype=self.C.dtype, device=self.C.device)
        self.prior = matrix_normal_inv_wishart(
            self.C, precision, degrees_of_freedom, self.Sigma)
        self.posterior = self.prior

    def attach(self, emission):
        emission.observation_policy = self
        emission.C = ObservationHistory(self, 'C', emission)
        emission.Sigma = ObservationHistory(self, 'Sigma', emission)
        emission.C_def = self.prior.get_mean()
        emission.Sigma_def = self.prior.get_scale()
        emission.observation_params = self.posterior
        variance = torch.diag(self.Sigma).sqrt().reshape(-1, 1)
        if emission.y_var:
            emission.y_var[-1] = variance
        else:
            emission.y_var = [variance]
        emission.gp.assign_alpha_ini(self.Sigma, emission.Gamma[0])

    def kl(self):
        if self.scope == 'fixed':
            return self.Sigma.new_zeros(())
        return self.posterior.kl_divergence(self.prior)

    def convert(self, target):
        """Convert the single owned posterior during model state conversion."""
        method = {'numpy': 'to_numpy', 'torch': 'to_torch',
                  'cpu': 'to_cpu', 'cuda': 'to_cuda'}[target]
        getattr(self.prior, method)()
        if self.posterior is not self.prior:
            getattr(self.posterior, method)()
        if target == 'numpy':
            self.C = self.posterior.get_mean()
            self.Sigma = self.posterior.get_scale(final=True)
        elif target == 'torch':
            self.C = torch.as_tensor(self.posterior.get_mean())
            self.Sigma = torch.as_tensor(self.posterior.get_scale(final=True))
        elif target == 'cpu':
            self.C = self.C.cpu()
            self.Sigma = self.Sigma.cpu()
        else:
            self.C = self.C.cuda()
            self.Sigma = self.Sigma.cuda()

    def update(self, emissions, responsibilities, *, online=False, current_index=None):
        """Apply one pooled conjugate update, counting the prior exactly once.

        Offline responsibilities have shape (beats, clusters) and recompute the
        coordinate optimum from the original prior. Online responsibilities
        have shape (clusters,) and add the new beat to the current posterior.
        """
        if self.scope != 'shared':
            return
        device, dtype = self.C.device, self.C.dtype
        observations, latents = [], []
        latent_cov = torch.zeros_like(self.Sigma)
        mass = 0.0
        for cluster, emission in enumerate(emissions):
            positions = [len(emission.indexes) - 1] if online else range(len(emission.indexes))
            for position in positions:
                if position < 0:
                    continue
                index = emission.indexes[position]
                if online and index != current_index:
                    continue
                weight = float(responsibilities[cluster] if online else responsibilities[index, cluster])
                if weight <= 0:
                    continue
                root = weight ** 0.5
                observation = emission.project_observation_to_basis(
                    emission.y_train[position], emission.x_train[position])
                observations.append(root * observation.to(device=device, dtype=dtype))
                latents.append(root * emission.f_star_sm[position + 1].reshape(-1, 1).to(device=device, dtype=dtype))
                latent_cov = latent_cov + weight * emission.cov_f_sm[position + 1].to(device=device, dtype=dtype)
                mass += weight
        if mass == 0:
            return
        zero = torch.zeros_like(self.Sigma)
        base = self.posterior if online else self.prior
        self.posterior = base.posterior(
            mass, torch.cat(observations, dim=1), torch.cat(latents, dim=1),
            zero, latent_cov, zero)
        self.C = self.posterior.get_mean()
        self.Sigma = self.posterior.get_scale(final=True)
        for emission in emissions:
            emission.observation_params = self.posterior
            if emission.y_var:
                emission.y_var[-1] = torch.diag(self.Sigma).sqrt().reshape(-1, 1)
