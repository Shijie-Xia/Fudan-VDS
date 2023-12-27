<div align="center">
  <h1>Fudan-SDS: Building Your Own Simulated Driving System</h1>
  <span style="color:red">📢 <strong><i>Play racing games like driving in the real world!!</i></strong></span>
</div>

<h2>🎉 Introduction</h2>

This is a project provided by Fudan University for fourth-year undergraduate students in the 'Intelligent System Design' course. Students are required to independently design their systems using *Nvidia Jetson Nano*. Here, our groups have developed a simulated driving system where users can control the game through actions similar to real-world driving using the development board or their own hardware!


## 👨‍🚀 Getting Started

1. **Clone the Repository and Navigate into the Directory** - Once your terminal is open, you can clone the repository and move into the directory by running the commands below.
   ```bash
   git clone https://github.com/Shijie-Xia/Simple-driving-simulation-system.git
   cd Simple-driving-simulation-system
   ```
2. **Install the enviroment** - You need to install the necessary library for the client and server. The **client** means the device where you play the game while the **server** capture and analyse your posture with the camera. In our project, we use a PC for the client and the Nvidia Jeson Nano with a camera for the server. Of course a PC can play both roles. *Note we provide the GPU version of mediapipe for Jetson Jetson Nano to improve the frame rate of image processing.* Now install the enviroment for them!
    ```bash
    # necessary library for the client
    pip install -r requirements_client.txt

    # necessary library for the server
    pip install -r requirements_server.txt

    # the GPU version of mediapipe for Jetson Jetson Nano
    pip install ./lib/mediapipe-0.10.7-cp38-cp38-linux_aarch64.whl
    ```
3. **Install and config Vjoy and Xoutput for the server** - To enable the client react to the information from the program by pyvjoy, it needs the Vjoy and Xoutput to convert it into a controller command. We provide detailed instruction for it in Chinese in folder `docs`. Follow it step by step.

4. **Run the program in server and client** - The first step is to start the server. Run the program. Make sure it is connected to the camera.
    ``` bash
    python ./code/server.py
    ```
    You will get an IP address of the server. Modify the corresponding part in `./code/client.py`.

    ```python
    # Modify the IP and PORT according to the server
    IP = 'the IP of the server'
    PORT = 'the port of the server'
    client.connect((IP, PORT))
    ```
    Run the program in client and open the computer game. Enjoy it!

    ```bash
    python ./code/client.py
    ```
