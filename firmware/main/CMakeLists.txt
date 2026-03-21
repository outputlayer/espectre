set(SRCS
    "main.c"
    "csi_collector.c"
    "stream_sender.c"
    "nvs_config.c"
    "ota_update.c"
)

idf_component_register(
    SRCS ${SRCS}
    INCLUDE_DIRS "."
)
