# -*- mode: ruby -*-
# vi: set ft=ruby :

$script = <<SCRIPT
apt-get update
apt-get install -y libcv-dev python-opencv
apt-get install -y python-numpy
SCRIPT

# Vagrantfile API/syntax version. Don't touch unless you know what you're doing!
VAGRANTFILE_API_VERSION = "2"

Vagrant.configure(VAGRANTFILE_API_VERSION) do |config|
  config.vm.box = "box-cutter/ubuntu1404-desktop"
  config.vm.hostname = "ultraking"

  config.vm.provision "shell", inline: $script
  
  config.vm.provider "virtualbox" do |vb|
    vb.gui = true
    vb.customize ['modifyvm', :id, '--usb', 'on']
  end
end
