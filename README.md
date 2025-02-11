# hackathon-fiap

## Preparação do ambiente

Execute esses seguintes comandos para subir seu ambiente local:

    python -m venv .venv 
    source .venv/bin/activate  # On macOS/Linux
    # .venv\Scripts\activate  # On Windows
    pip install -r requirements.txt 

## Configurando seu dispositivo para notificação

1 - Visite o site pushbullet.com, faça o login
2 - Após disso vá em Set up your phone ou clique em Set up your computer caso queira configurar no seu note
3 - Agora vá em My Account, na parte de accessTokens gere sua chave e guarde esse valor
4 - Com a chave gerada realize os seguintes comandos no seu terminal de preferencia:

```bash
    echo "export PUSHBULLET_API_KEY='YOUR_API_KEY'" > pushbullet.env
    echo "pushbullet.env" >> .gitignore
    source ./pushbullet.env
```

### Para acessar no celular é necessário realizar esses seguintes passos:
1 - Faça o download do app PushBullet, inicie a sessão
2 - Habilite a configuração de notificação no dispositivo

### Para acessar no note é necessário realizar esses seguintes passos
1 - Instalar e configurar extensão do PushBullet


## Instruções de execução do algoritmo

1 - Faça o download do dataset em: [link](https://drive.google.com/drive/folders/1PijOS0mtrFbyHxNMU8WqcOK_1uHAzX_O?usp=drive_link)

E coloque as pastas "cortante" e "nao_cortante" dentro da pasta "data"

2 - Para executar o arquivo main, primeiro você deve executar o arquivo modelo.py para gerar o modelo na pasta "modelo/"

3 - Após disso, faça o download desses dois videos abaixo para poder executar o main.py

- [Video 1](https://drive.google.com/file/d/1AV6y7OFPgq9UiU0TMUjoaoYQHsvKO__u/view?usp=sharing)

- [Video 2](https://drive.google.com/file/d/1XBhBKY9QHo0xj8gXMYcq92e-vrECrNH3/view?usp=sharing)

Lembre-se de colocar o endereço do arquivos de video corretos na linha 15 do arquivo main.py

Com esse setup você já deve conseguir executar o código do desafio desse Hackathon

