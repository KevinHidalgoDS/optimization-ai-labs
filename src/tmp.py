#%%

from src.utils import logger as log

logger = log.LOGGER

config = log.CONFIG

#%%
logger.info("hello world")
logger.info("variables de configuracion:\n%s", config)

config.paths.data.raw