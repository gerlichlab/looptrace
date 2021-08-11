import socket
import threading
import sys
import time

def socket_server():

    host = 'localhost'
    port = 65432
    address = (host, port)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(address)
    server_socket.listen(5)

    while True:
        print("Listening for client . . .")
        conn, address = server_socket.accept()
        print("Connected to client at ", address)
        #pick a large output buffer size because i dont necessarily know how big the incoming packet is                                                    
        while True:
            data = conn.recv(2048);
            if data.strip() == b"disconnect":
                conn.close()
                print('Closing connection.')
                break
                #sys.exit("Received disconnect message.  Shutting down.")
                #conn.send(b"dack")
            elif data.strip() == b"shutdown":
                conn.close()
                print('Received shutdown message.  Shutting down.')
                sys.exit()

            elif data:
                print("Message received from client:")
                print(data)
                conn.send(b"Replying.")


t1 = threading.Thread(target=socket_server)
t1.start()
