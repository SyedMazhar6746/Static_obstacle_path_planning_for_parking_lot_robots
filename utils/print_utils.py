#!/usr/bin/env python3


from colorama import Fore, Style 

"""Print functions for different colors"""    

def print_green(message, value, color=Fore.GREEN, style=Style.BRIGHT):
    print(color + style + message,  value,  Style.RESET_ALL) 

def print_green_text(message, color=Fore.GREEN, style=Style.BRIGHT):
    print(color + style + message, Style.RESET_ALL) 
 
def print_red(message, value, color=Fore.RED, style=Style.BRIGHT): 
    print(color + style + message,  value,  Style.RESET_ALL) 

def print_red_text(message, color=Fore.RED, style=Style.BRIGHT):
    print(color + style + message + Style.RESET_ALL) 

def print_yellow(message, value, color=Fore.YELLOW, style=Style.BRIGHT):
    print(color + style + message,  value,  Style.RESET_ALL) 

def print_yellow_text(message, color=Fore.YELLOW, style=Style.BRIGHT):
    print(color + style + message + Style.RESET_ALL) 

def print_magenta_line(color=Fore.MAGENTA, style=Style.BRIGHT):
    message = "=====================================================================================================" 
    print(color + style + message + Style.RESET_ALL) 

