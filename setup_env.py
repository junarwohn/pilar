import configparser

def create_config():
    config = configparser.ConfigParser()
    
    # Get credentials from user input
    print("Please enter your credentials:")
    url = input("URL (default: https://example.com): ").strip() or "https://example.com"
    user_id = input("ID: ").strip()
    password = input("Password: ").strip()
    
    # Add credentials section
    config['Credentials'] = {
        'url': url,
        'id': user_id,
        'password': password
    }
    
    # Write to config.ini file
    with open('config.ini', 'w') as configfile:
        config.write(configfile)
        print("\nConfig file created successfully!")

if __name__ == '__main__':
    create_config()
