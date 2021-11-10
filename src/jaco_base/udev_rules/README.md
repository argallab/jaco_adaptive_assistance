Add these udev rules to /etc/udev/rules.d
66-tty.rules gives permissions to the arduinos for the chin switch and mode switch display. 60-usb-serial.rules gives the alternate name /dev/sip_and_puff to the sip and puff, 3-axis joystick and /dev/ps3 to the ps3 controller instead of /dev/input/js*.

###A note on writing udev rules for giving static id's to joystick device input:

Use the following command to find information about the device that can be used for writing udev rules for /dev/input devices (like joystick or mouse):

sudo udevadm info --name=/dev/input/js* --attribute-walk 
or for tty:

 sudo udevadm info --attribute-walk --name /dev/ttyACM*
subsitute * with the number of current device. If the device is a mouse, then it would be /dev/input/mouse*, if it is joystick /dev/input/js*, etc.

Input enough unique identifiers in the .rules file. From experience, I have found that for giving permissions, idVendor and idProduct are enough. However, for making a simlink KERNEL, SUBSYSTEM, and SUBSYSTEMS are also required.

After saving the file (need root permissions), re-load the rules using:

sudo udevadm control --reload-rules && udevadm trigger
Check to make sure the symlink has been made correctly. For example, if the simlink name is now /dev/ps3 instead of /dev/input/js1, you can check if the symlink was succesful:

ls -l /dev/sip_and_puff
If this outputs something like this:

lrwxrwxrwx 1 root root 9 May 30 12:47 /dev/ps3 -> input/js1
Then everything will work. Pay special attention to ->input/js1. If instead it is linking to something else (e.g ->input/event) than you don't have enough/correct unique identifiers in the .rules file.