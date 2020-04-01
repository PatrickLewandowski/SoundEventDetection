import socket

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

host = '127.0.0.1' #socket.gethostname()
port = 7634

s.connect((host,port))

messag = input("Enter message:")
s.send(messag.encode())

s_message = s.recv(1024)
print(s_message.decode())

s.close()