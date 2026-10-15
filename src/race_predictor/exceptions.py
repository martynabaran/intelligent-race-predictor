"""Custom exceptions."""


class RacePredictorError(Exception):
    """Base class for all package errors."""


class UnsupportedExportFormatError(RacePredictorError):
    """The given file is not a recognised Strava / Garmin export."""


class DataValidationError(RacePredictorError):
    """Input data is missing required columns or is otherwise unusable."""
