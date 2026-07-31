"""
tools/crypto_utils.py
----------------------
Shared at-rest encryption for any "connected account" secret this app
stores on a user's behalf -- currently Gmail's OAuth refresh token
(models.GmailConnection) and Slack's incoming webhook URL
(models.SlackConnection). Both are equivalent to a password for their
one scope (anyone holding a Slack webhook URL can post to that
channel; anyone holding the Gmail refresh token can send mail as that
user), so both get encrypted the same way rather than stored plaintext.

Originally this lived inline in tools/emailer.py as Gmail-specific
helpers; factored out here once Slack needed the identical mechanism,
rather than copy-pasting a second Fernet setup.

GMAIL_TOKEN_ENCRYPTION_KEY is the one encryption key covering ALL such
secrets, not just Gmail's -- the env var name predates Slack support
and is kept as-is rather than renamed, to avoid breaking anyone who
already generated and deployed one.
"""

import os
from cryptography.fernet import Fernet, InvalidToken


class SecretDecryptError(Exception):
    """Raised when a stored secret can't be decrypted -- almost always
    means the encryption key changed since the secret was stored.
    Callers should treat this as 'ask the user to reconnect', not a
    hard crash."""


def _get_fernet() -> Fernet:
    """Must be a real Fernet key (32 url-safe base64-encoded bytes) --
    generate one once with:
        python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    and store it in .env as GMAIL_TOKEN_ENCRYPTION_KEY. Deliberately
    fails loudly rather than falling back to a default key, since a
    guessable/shared key would make every stored secret decryptable by
    anyone with the code -- same reasoning as DATABASE_URL's fail-fast
    in web_app.py."""
    key = os.environ.get("GMAIL_TOKEN_ENCRYPTION_KEY")
    if not key:
        raise RuntimeError(
            "GMAIL_TOKEN_ENCRYPTION_KEY is not set. Generate one with:\n"
            "  python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"\n"
            "and add it to your .env file. This key encrypts every stored "
            "connected-account secret (Gmail refresh tokens, Slack webhook "
            "URLs) at rest -- there is no insecure fallback."
        )
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt_secret(raw_value: str) -> str:
    return _get_fernet().encrypt(raw_value.encode()).decode()


def decrypt_secret(encrypted_value: str) -> str:
    try:
        return _get_fernet().decrypt(encrypted_value.encode()).decode()
    except InvalidToken:
        raise SecretDecryptError(
            "Could not decrypt a stored secret -- the encryption key may "
            "have changed. The connected account needs to be reconnected."
        )