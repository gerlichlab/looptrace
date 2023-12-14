cat <<EOF >/etc/apt/preferences.d/proposed-updates
# Configure apt to allow selective installs of packages from proposed
Package: *
Pin: release a=$(lsb_release -cs)-proposed
Pin-Priority: 400
EOF