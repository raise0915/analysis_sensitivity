import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Email details
sender_email = "yuri1154.sg@gmail.com"
receiver_email = "yuri1154.sg@gmail.com"
password = os.environ.get('EMAIL_PASSWORD')


def create_message(subject, body):
    # Create a MIMEText object
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    return msg


def send_email(subject, body):
    # Send the email
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        msg = create_message(subject, body)
        msg['From'] = sender_email
        msg['To'] = receiver_email

        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()

    except Exception as e:
        print(f"Failed to send email: {e}")
