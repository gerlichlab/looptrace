cat <<EOF >/etc/apt/sources.list.d/ubuntu-$(lsb_release -cs)-proposed.list
# Enable Ubuntu proposed archive
deb http://archive.ubuntu.com/ubuntu/ $(lsb_release -cs)-proposed restricted main multiverse universe
EOF