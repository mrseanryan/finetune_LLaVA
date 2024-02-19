# see https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md
set -e -x

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
