# echo "export SENDGRID_API_KEY='key" > sendgrid.env
# echo "sendgrid.env" >> .gitignore
# source ./sendgrid.env

# pip install sendgrid
# https://github.com/sendgrid/sendgrid-python


# using SendGrid's Python Library
# https://github.com/sendgrid/sendgrid-python
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

message = Mail(
    from_email='riosistemas@riosistemas.com.br',
    to_emails='nesralla@gmail.com',
    subject='[FIAP] Imagem detectada',
    html_content='<strong>Aqui esta a notificacao porque foi detectada um objeto suspeito</strong>')
try:
    sg = SendGridAPIClient('A CHAVE ESTA NO GRUPO DO WHATSAPP')
    response = sg.send(message)
    # print(response.status_code)
    # print(response.body)
    # print(response.headers)
except Exception as e:
    print(e.message)