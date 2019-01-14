#!/usr/bin/env python3
"""
no-ig-relogin.py:
	Download with InstaLooter without relogging for every user.

Usage:
	python no-ig-relogin.py <output_dir> <user_list>

Positional Arguments:
	output_dir		the directory where images are downloaded
					(in subdirectories named after each user)
	user_list		the path to the file containing the usernames
					from which to download pictures (one username
					per line).
"""

import instaLooter
import os
import sys
import getpass



# Just give instaLooter the first profile so that it
# understands it needs to download pictures from a
# profile page.
looter = instaLooter.InstaLooter(profile='kimkardashian', get_videos=True)




looter.target = 'kimkardashian'
looter.directory = '/mnt/CABC33CABC33B035/instagram_loot'

# Create the directory if it does not exist
if not os.path.isdir(looter.directory):
    os.makedirs(looter.directory)

# Download new files only, with the progress bar
looter.download(new_only=True, with_pbar=True)