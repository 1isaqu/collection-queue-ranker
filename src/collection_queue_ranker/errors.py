"""Exception types. Every error the package raises on purpose derives from CqrError."""


class CqrError(Exception):
    """Base class for expected, user-facing failures."""


class ConfigError(CqrError):
    """A settings file or environment variable is missing, unknown or malformed."""


class ContractError(CqrError):
    """A source is missing a table or a required column of the data contract."""


class UnsafeSqlError(CqrError):
    """A statement was refused because it is not a plain read-only SELECT."""


class DataError(CqrError):
    """The data cannot support the requested step (empty frame, single class, bad window)."""


class PipelineError(CqrError):
    """A stage of the retrain pipeline failed; later stages were deliberately not run."""

    def __init__(self, stage: str, cause: Exception) -> None:
        super().__init__(
            f"stage `{stage}` failed: {cause}. Later stages were not run, so nothing downstream "
            "was regenerated from stale or partial data."
        )
        self.stage = stage
        self.cause = cause
