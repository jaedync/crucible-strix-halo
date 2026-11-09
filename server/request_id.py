"""The name of the request header that carries the gateway's request id. A gateway in front of the server may send its
own id under any header name; the operator names it here. The value rules stay in metrics.client_request_id."""
import re

CLIENT_REQUEST_ID_HEADER_ENV = "CRUCIBLE_CLIENT_REQUEST_ID_HEADER"
DEFAULT_CLIENT_REQUEST_ID_HEADER = "X-Client-Request-Id"
# An HTTP token of letters, digits and hyphens: a name that no header can fail to carry.
HEADER_NAME = re.compile(r"[A-Za-z0-9-]{1,64}")


# Headers that carry a credential. The value of the id header goes to the log lines (a 401 line names it), so one of
# these would write the key that a client tried, or the valid key of a gateway, to the journal.
CREDENTIAL_HEADERS = frozenset({"authorization", "proxy-authorization", "x-api-key", "cookie", "set-cookie"})


class HeaderNameError(ValueError):
    pass


def resolve_client_request_id_header(flag, environ):
    """The header name: the flag wins over the environment variable, and an unset or empty variable is the default.

    A flag that is given but wrong (including an empty one) is an error; it must not fall back to another source."""
    if flag is not None:
        source, name = "--client-request-id-header", flag
    else:
        source, name = CLIENT_REQUEST_ID_HEADER_ENV, environ.get(CLIENT_REQUEST_ID_HEADER_ENV) or DEFAULT_CLIENT_REQUEST_ID_HEADER
    if not HEADER_NAME.fullmatch(name):
        raise HeaderNameError(f"{source}: {name[:40]!r} is not a header name (1 to 64 letters, digits or hyphens)")
    if name.lower() in CREDENTIAL_HEADERS:
        raise HeaderNameError(f"{source}: {name!r} carries a credential, and the id is written to the log. "
                              "Name another header, for example X-Request-Id.")
    return name
