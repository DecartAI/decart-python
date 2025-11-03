"""User-Agent header construction for SDK requests."""

from typing import Optional
from ._version import __version__


def build_user_agent(integration: Optional[str] = None) -> str:
    """
    Builds the User-Agent string for the SDK.
    
    Format: decart-python-sdk/{version} lang/py {integration?}
    
    Args:
        integration: Optional integration identifier (e.g., "langchain/0.1.0")
    
    Returns:
        Complete User-Agent string
    
    Examples:
        >>> build_user_agent()
        'decart-python-sdk/0.0.6 lang/py'
        
        >>> build_user_agent("langchain/0.1.0")
        'decart-python-sdk/0.0.6 lang/py langchain/0.1.0'
    """
    parts = [f"decart-python-sdk/{__version__}", "lang/py"]
    
    if integration:
        parts.append(integration)
    
    return " ".join(parts)
