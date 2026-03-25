import ad_tools.tools as tools
import wandb

sweep_configuration = {"method": "bayes",
                       "metric" : {"goal": "maximize",
                                    "name": "hs_kl_signal_acceptance_rate"},
                        "parameters": {"beta_upper_limit": {"distribution": "uniform","min": 0, "max": 0.5}}}
project_name = "BetaVAEMark3 Tuning2"
sweep_id = wandb.sweep(sweep= sweep_configuration, project = project_name)

train_function = lambda input = None: tools.tune(project_name= project_name, model= tools.BetaVAEMark3(), lr = 1e-4)

wandb.agent(sweep_id, function = train_function, count = 100)