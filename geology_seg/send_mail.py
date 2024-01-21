import time
import os
import shutil
import smtplib
from email.mime.text import MIMEText
import argparse

def send_mail(msg, title, passwd, sender='719751595@qq.com', accepters=['719751595@qq.com', ]):#'554872480@qq.com'
    message = MIMEText(msg,'plain','utf-8')
    message['Subject'] = title
    message['From'] = sender
    
    smtpObj = smtplib.SMTP_SSL('smtp.qq.com')
    #smtpObj.connect('smtp.smail.nju.edu.cn')
    smtpObj.login(sender.split('@')[0], passwd)
    for a in accepters:
        message['To'] = a
        smtpObj.sendmail(sender, a, message.as_string())
    smtpObj.quit()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--no", type=int,default=0)
    parser.add_argument("--passwd", type=str,default=None)
    opts = parser.parse_args()
    
    assert opts.passwd is not None
    
    now = int(round(time.time()*1000))
    format_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(now/1000+28800))
    with open('./val'+str(opts.no)+'.txt', 'r') as f:
        send_mail(f.read(),format_time,opts.passwd)
    #with open('./temp_val.txt', 'w') as f:
    #    f.write(format_time)
    if not os.path.exists('../val_texts'):
        os.mkdir('../val_texts')
    os.rename('./val'+str(opts.no)+'.txt', './val'+str(opts.no)+'_'+'_'.join(format_time.split(' '))+'.txt')
    shutil.move('./val'+str(opts.no)+'_'+'_'.join(format_time.split(' '))+'.txt', '../val_texts')
    print(format_time)
        
        