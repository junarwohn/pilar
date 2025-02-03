import configparser
import os

def create_config():
    config = configparser.ConfigParser()
    
    # Get credentials from user input
    print("Starting environment setup...")
    print("\nAutomatic Upload Configuration:")
    url = input("Enter upload URL: ").strip()
    user_id = input("Enter ID: ").strip()
    password = input("Enter password: ").strip()
    
    # Chrome configuration
    print("\nChrome Configuration:")
    driver_path = input("Enter Chrome driver path (or press Enter for default): ").strip()
    user_data_dir = input("Enter Chrome user data directory (or press Enter for default): ").strip()
    profile_directory = input("Enter Chrome profile directory (or press Enter for default): ").strip()
    
    # Set default values if empty
    if not driver_path:
        driver_path = "chromedriver"
    if not user_data_dir:
        # if linux  
        if os.name == 'posix':
            user_data_dir = os.path.expanduser('~/.config/google-chrome')
        # if windows
        else:
            user_data_dir = os.path.expanduser('~') + r'\AppData\Local\Google\Chrome\User Data'
    if not profile_directory:
        profile_directory = "Default"
    
    # Add credentials section
    config['Credentials'] = {
        'url': url,
        'id': user_id,
        'password': password,
        'driver_path': driver_path,
        'user_data_dir': user_data_dir,
        'profile_directory': profile_directory
    }
    
    # Write to config.ini file
    with open('config.ini', 'w', encoding='utf-8') as configfile:
        config.write(configfile)
        print("\nConfiguration saved successfully!")

if __name__ == '__main__':
    create_config()
