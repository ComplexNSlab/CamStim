# Prerequisites

Ensure that you either <a href="https://www.anaconda.com/download">install Anaconda</a> or <a href="https://docs.anaconda.com/free/miniconda/miniconda-install/">install miniconda</a> on your computer.

You can verify that you have Anaconda if you have Anaconda Navigator in your applications. You can also verify by running the command ```conda --version``` in the Terminal (Mac/Linux) or the Anaconda Prompt application (Windows).

# Environment Setup

1. Click the green 'Code' button at the top of this page and select "Download Zip." This will download "speckle-imaging-main.zip" onto your computer. Unzip this folder and move it to your desired location in your file directory. 
2. In your Terminal/Anaconda Prompt, navigate to the location of your unzipped folder.
>NOTE: use the commands ```ls``` to see subdirectories and ```cd``` <subdirectory> to navigate through your files. You can use cd .. to go to the parent directory.

3. Enter this command:
    ```
    conda create -n fb-cam python=3.8
    ```
# Activating the Environment

4. Within the unzipped folder, you should find a file titled "fb-cam.py". You will need the path of this file later, so copy the path onto your clipboard.
1. Activate the environment with ```conda activate fb-cam```

2. Within the unzipped folder, you should find a file titled "fb-cam.py". You will need the path of this file later, so copy it onto your clipboard.
> Mac: right click on the file to open the context menu, then hold down the option key to reveal additional options. Click on 'Copy "fb-cam.py" as Pathname" to copy the path to your clipboard.

> Windows: right click on the file to open the context menu, then click 'Copy as path'.

5. Open the Anaconda Prompt application on your computer and enter the following:
    ```
    conda activate fb-cam
    ```
6. Copy the filepath and paste it in the terminal and press [ENTER]:
    ```
    python <PATH COPIED>
    ```
This should start the program as well as open a new window showing the input through your lens.

# Keyboard Controls

Use the following keyboard commands to navigate + change settings within the program. 
To enter a command, click on the window displaying the lens output before typing your key. If additional follow up input is required (eg new exposure or gain numbers), click back to the terminal window before typing.

q -- quit<br>
t -- switch camera mode to hardware trigger capture<br>
c -- switch camera mode to continuous capture<br>
s -- starts saving captured frames. Press again to stop capturing<br>
e -- edit exposure<br>
g -- edit gain<br>
p -- print camera stats<br>
m -- start pwm<br>
n -- stop pwm<br>
x -- start experiment mode. Press again to stop experiment mode.<br>

To display these commands in terminal, press h