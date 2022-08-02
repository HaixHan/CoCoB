import smtplib
from email.mime.text import MIMEText

def send_email():
    From = 'XX@qq.com'
    To = From
    pwd = 'qfmnvxbudzzijbaf'
    # login
    smtp = smtplib.SMTP()
    smtp.connect('smtp.qq.com')
    smtp.login(From, pwd)
    # email
    mail = MIMEText('''My master,the training of the NewIdea has been completed, please check.''')
    mail['Subject'] = 'Progress of training'
    mail['From'] = From
    mail['To'] = To
    # send
    smtp.sendmail(From, To, mail.as_string())
    print('send email success!!!!')
    smtp.quit()
