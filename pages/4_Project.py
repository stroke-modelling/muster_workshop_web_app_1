import streamlit as st

from utilities.utils import page_setup, write_text_from_file

page_setup()

write_text_from_file('pages/text_for_pages/4_Project.txt',
                     head_lines_to_skip=2)
