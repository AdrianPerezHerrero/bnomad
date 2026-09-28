# 0.3 migration inventory

Inventory completed before changes to executable files. `inventory.json` contains every definition and every Python/notebook textual use with file/line and source. It distinguishes occurrences (including comments) without treating each as an executable import. All root model methods have an assigned owner.

| Legacy symbol | Responsibility | New equivalent | Fully migrated? | Used by | Safe to remove? |
|---|---|---|---|---|---|
| `GPI_HDP.__init__` | model/_state.py | model/_state.py | no | See inventory.json line-level reference list | only after callers migrate |
| `GPI_HDP.keep_last_all`, `set_default_options`, `get_default_options`, `create_gp_default`, `create_wp_sys_default`, `refill`, `reorder`, `new_group`, `remove_last_group`, `redefine_default`, `update_initial_sigma`, `return_model_of_sample`, `selected_gpmodels`, `gpmodel_deepcopy` | model/_clusters.py | model/_clusters.py | no | See inventory.json line-level reference list | only after callers migrate |
| `GPI_HDP._safe_exp`, `variational_local_terms`, `LogLik`, `forward`, `backward`, `coupled_state_coef`, `sum_coupled_state_coef`, `compute_h`, `baum_welch`, `normalize_log` | inference/_messages.py | inference/_messages.py | no | See inventory.json line-level reference list | only after callers migrate |
| `GPI_HDP.signaltonoise`, `rolling_snr`, `weight_mean`, `reduce_num_outputs`, `compute_snr_ini`, `compute_snr`, `normalize_snr`, `compute_joint_xy_q` | emissions/_weights.py | emissions/_weights.py | no | See inventory.json line-level reference list | only after callers migrate |
| `GPI_HDP.compute_q_elbo`, `full_LDS_elbo`, `estimate_new`, `compute_q` | inference/_scoring.py | inference/_scoring.py | no | See inventory.json line-level reference list | only after callers migrate |
| `GPI_HDP.cond_to_torch`, `cond_to_numpy`, `cond_cuda`, `cond_cpu`, `cond_detach`, `full_model_to_numpy`, `full_model_to_torch`, `full_model_to_cuda`, `full_model_to_cpu`, `save_swgp` | model/_conversion.py | model/_conversion.py | no | See inventory.json line-level reference list | only after callers migrate |
| `GPI_HDP.include_sample`, `include_sample_fast`, `_init_resp_cache_cold_start`, `_ensure_fast_cache_online`, `_update_q_lat_tail`, `_append_hard_step` | inference/_online.py | inference/_online.py | yes; forwarding only | See inventory.json line-level reference list | only after callers migrate |
| `GPI_HDP.include_batch`, `variational_local_terms_batch`, `estimate_q_first`, `estimate_q_all`, `cluster_new_batch`, `reload_model_from_labels` | inference/_offline.py | inference/_offline.py | yes; forwarding only | See inventory.json line-level reference list | only after callers migrate |
| `GPI_HDP.reset_warp_cache`, `compute_warp_actual_state`, `compute_warp_actual_state_amtgp`, `compute_warp_y`, `warp_batch_by_resp_amtgp`, `warp_batch_by_resp_amtgp_cached`, `select_assigned_warp` | alignment/_components.py | alignment/_components.py | yes; forwarding only | See inventory.json line-level reference list | only after callers migrate |
| `GPI_HDP.init_global_params`, `reinit_global_params`, `temp_reinit_global_params`, `create_initrho`, `_calcThetaPost`, `_calcThetaFull`, `compute_Pi`, `rho2beta`, `beta2rho`, `elbo_Linears`, `refill_resp`, `calcELBO_LinearTerms`, `calcELBO_NonlinearTerms`, `calc_Hstart`, `calc_Htable`, `L_top`, `c_Dir`, `find_optimum_rhoOmega`, `compute_trans_A`, `compute_trans_pi` | hdp/_variational.py | hdp/_variational.py | yes; forwarding only | See inventory.json line-level reference list | only after callers migrate |
| `GPI_HDP.from_config`, `hdp` | remove compatibility | remove compatibility | no | See inventory.json line-level reference list | only after callers migrate |
| `GPI.IterativeGaussianProcess` | GP covariance, projection, predictions, optimization | `emissions/_prior.py` | no | inventory.json | after caller migration |
| `GPI_model.GPI_model` | Emission trajectories and GP/LDS integration | `emissions/_gaussian_process.GaussianProcessEmission` | no | inventory.json | after caller migration |
| `GPI_models_pytorch` | Active Exact/Projected/VarProjected GP models | `emissions/_torch_models.py` | no | inventory.json | after caller migration |
| `amtgp_warping_system.Warping_system` | Current AMTGP warping | `alignment/_amtgp.GaussianProcessWarp` | no | inventory.json | after caller migration |
| `OptimizerRhoOmega` | bnpy optimizer re-exports | `hdp/_optimizer.py` | no | inventory.json | after caller migration |
| `warping_system.Warping_system` | Inactive alternative warp algorithm | `Retire; current supported alignment is AMTGP` | no | inventory.json | after caller migration |
| `util_plots` | Research visualization and verbose kernel reporting | `diagnostics/_plots.py` | no | inventory.json | after caller migration |
| `get_data` | Dataset loading, SNR and LDS initialization estimates | `datasets/_research.py` | no | inventory.json | after caller migration |
| `extract_data, GPI_test` | Historical side-effect experiment programs | `Retire in favor of public examples/tutorials` | no | inventory.json | after caller migration |

Duplicate: create_wp_sys_default is defined twice; only the latter is executable. Existing HDP, alignment and inference wrappers forward to single canonical implementations. Emission alias/shim mutates __module__ for old pickle paths. No independent new implementation exists behind public HDPGPC yet.

The five historical study scripts and two root research notebooks use the private old API and external/path-dependent datasets; maintained public tutorials replace these programs in this release. Preserve their originals in the unchanged 0.2 export. Optional dataset readers/plotters retain their algorithms at canonical private paths. Retired alternative warping and its unused GPyTorch classes have no supported public entry point.
