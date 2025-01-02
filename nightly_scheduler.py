def send_email():
    sender_email = "maxwell.b.resnick@gmail.com"
    receiver_email = "maxwell.b.resnick@gmail.com"
    password = "JohnTavares91"
    
    subject = "Script Status"
    body = "The script ran!"
    
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    
    message.attach(MIMEText(body, "plain"))
    
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())
            print("Email sent!")
    except Exception as e:
        print(f"Failed to send email: {e}")

def job():
    os.system('caffeinate -s python /Users/maxwellresnick/georgia/data_etl.py')
    send_email()

schedule.every().day.at("09:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(60)