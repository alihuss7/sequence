"""Contact Us page content."""

import os
import smtplib
from email.message import EmailMessage
from email.utils import formataddr

import streamlit as st


def _truthy(value, default=False):
    """Return bool for truthy string values."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _load_smtp_settings():
    """Load SMTP settings from Streamlit secrets or environment variables."""
    try:
        secrets_block = st.secrets.get("smtp")
    except Exception:
        secrets_block = None

    if secrets_block:
        return dict(secrets_block)

    env_map = {
        "host": os.environ.get("SMTP_HOST"),
        "port": os.environ.get("SMTP_PORT"),
        "username": os.environ.get("SMTP_USERNAME"),
        "password": os.environ.get("SMTP_PASSWORD"),
        "sender": os.environ.get("SMTP_SENDER"),
        "sender_name": os.environ.get("SMTP_SENDER_NAME"),
        "recipient": os.environ.get("SMTP_RECIPIENT"),
        "use_ssl": os.environ.get("SMTP_USE_SSL"),
        "use_tls": os.environ.get("SMTP_USE_TLS"),
    }

    if any(env_map.values()):
        return {k: v for k, v in env_map.items() if v is not None}

    return None


def _send_notification_email(name: str, email: str, message: str) -> None:
    """Send email notification for contact submissions using SMTP secrets."""
    smtp_settings = _load_smtp_settings()
    if not smtp_settings:
        raise RuntimeError(
            "Missing SMTP credentials. Provide .streamlit/secrets.toml or environment variables."
        )

    host = smtp_settings.get("host")
    port_value = smtp_settings.get("port", 587)
    port = int(port_value or 587)
    username = smtp_settings.get("username")
    password = smtp_settings.get("password")
    sender_email = smtp_settings.get("sender", username)
    sender_name = smtp_settings.get("sender_name", "Sequence Contact Form")
    recipient_email = smtp_settings.get("recipient", "hussainjr.ali@gmail.com")
    use_ssl = _truthy(smtp_settings.get("use_ssl"), False)
    use_tls = _truthy(smtp_settings.get("use_tls"), not use_ssl)

    if not all([host, port, username, password, sender_email, recipient_email]):
        raise RuntimeError("Incomplete SMTP settings; check host/port/credentials")

    msg = EmailMessage()
    msg["Subject"] = "New contact form submission"
    msg["From"] = formataddr((sender_name, sender_email))
    msg["To"] = recipient_email
    msg.set_content(
        f"""
New form submission received.

Name: {name or 'N/A'}
Email: {email or 'N/A'}

Message:
{message or 'N/A'}
""".strip()
    )

    if use_ssl:
        with smtplib.SMTP_SSL(host, port) as server:
            server.login(username, password)
            server.send_message(msg)
    else:
        with smtplib.SMTP(host, port) as server:
            if use_tls:
                server.starttls()
            server.login(username, password)
            server.send_message(msg)


def render():
    """Render the contact us page."""
    st.header("Contact Us")

    with st.form("contact_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        message = st.text_area("Message")
        submit = st.form_submit_button("Send")

        if submit:
            with st.spinner("Sending your message..."):
                try:
                    _send_notification_email(name, email, message)
                except Exception as exc:  # pragma: no cover - user feedback path
                    error_text = str(exc)
                    if (
                        "No secrets found" in error_text
                        or "Missing [smtp]" in error_text
                        or "Missing SMTP" in error_text
                    ):
                        st.error("Email notifications are not configured yet.")
                        st.info(
                            "Add SMTP credentials to .streamlit/secrets.toml or set environment "
                            "variables (see README section 8) to enable contact email alerts."
                        )
                        st.code(
                            """[smtp]\nhost = \"smtp.gmail.com\"\nport = 587\nusername = \"your-gmail@example.com\"\npassword = \"app-password\"\nsender = \"your-gmail@example.com\"\nsender_name = \"Sequence Contact\"\nrecipient = \"hussainjr.ali@gmail.com\"\nuse_tls = true""",
                            language="toml",
                        )
                        st.caption(
                            "Environment option: set SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, "
                            "SMTP_SENDER (optional), SMTP_SENDER_NAME, SMTP_RECIPIENT, SMTP_USE_TLS/SMTP_USE_SSL."
                        )
                    else:
                        st.error(f"Failed to send email: {error_text}")
                else:
                    st.success("Thank you for your message! We've notified the team.")
