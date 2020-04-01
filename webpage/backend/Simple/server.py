import socket

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

host = '127.0.0.1' #socket.gethostname()
port = 7634

s.bind((host,port))
s.listen(1)

while True:
    con, addr = s.accept()
    print("connected with", addr)

    s_message = con.recv(1024)
    print(s_message.decode())

    #messag = [1,2,3,4,5,6,7,8,9]
    
    messag = s_message.decode()
    con.send(messag.encode())