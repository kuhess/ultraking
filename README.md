# Ultraking

OpenCV + Python implementation of the paper [Incremental Learning for Robust Visual Tracking](http://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf)

## Dependencies

- a webcam :movie_camera:
- Python 2.7
- OpenCV 2.4.x
- python-opencv

As described further, you can either use the virtual machine provided by Vagrant (recommended) or install manually all the dependencies.

## Play around

### Launch the pre-configured virtual machine

1. Install Virtualbox and the extension pack (needed for the webcam): https://www.virtualbox.org/wiki/Downloads
2. Install Vagrant: https://www.vagrantup.com/downloads.html
3. Start the pre-configured graphical VM with:

	```sh
	vagrant up
	```

4. Attach your webcam to the VM through the Virtualbox menu `Devices > Webcams`.
5. Open a terminal in the VM and go to the shared folder:

	```sh
	cd /vagrant
	```

### Run Ultraking

```sh
python ultraking.py
```

### Usage

1. Select the part of the image you want to track with the mouse (left click and drag)
2. Press `Enter` to start the tracking
3. Have fun
4. Quit by pressing a button (all except `Enter`)

## Resources

*Incremental Learning for Robust Visual Tracking*, http://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf
