from gymnasium.envs.registration import register

register(
    id="standard_cell_layout/StdCellPlaceEnv-v0",
    entry_point="standard_cell_layout.envs:StdCellPlaceEnv",
)
