import smtplib
from email.mime.text import MIMEText


def send_alert(patient_id, risk_score, provider_email):
    subject = f"High Cancer Risk Detected for Patient {patient_id}"
    body = f"Risk Score: {risk_score}. Immediate screening is recommended."

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = 'no-reply@oncoai.com'
    msg['To'] = provider_email

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login('your_email@gmail.com', 'your_password')
            server.sendmail('your_email@gmail.com', provider_email, msg.as_string())
        print(f"Alert sent to {provider_email}")
    except Exception as e:
        print(f"Failed to send alert: {str(e)}")
