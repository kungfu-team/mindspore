# Download and build build kungfu

INCLUDE(FetchContent)

FETCHCONTENT_DECLARE(
    kungfu
    GIT_REPOSITORY https://github.com/lsds/KungFu.git
    GIT_TAG master) # TODO: use a stable tag

FETCHCONTENT_POPULATE(kungfu)

# TODO: build kungfu here.
