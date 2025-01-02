import pandas as pd
import os
import schedule
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
from ftplib import FTP, error_perm
from io import BytesIO
from datetime import datetime

def download_and_concat_csv_files(ftp_host, ftp_user, ftp_password, remote_dirs, start_date=None):
    ftp = FTP(ftp_host)
    ftp.login(ftp_user, ftp_password)

    if isinstance(remote_dirs, str):
        remote_dirs = [remote_dirs]

    if start_date:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')

    def extract_date_from_filename(filename):
        match = re.match(r'^(\d{8})', filename)
        if match:
            return datetime.strptime(match.group(0), '%Y%m%d')
        
        match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
        if match:
            return datetime.strptime(match.group(0), '%Y-%m-%d')
        
        return None

    def traverse_ftp_directory(current_dir):
        ftp.cwd(current_dir)

        items = []
        ftp.retrlines('LIST', items.append)

        dfs = []

        for item in items:
            parts = item.split()
            item_name = parts[-1]

            if item_name in ['.', '..']:
                continue

            if parts[0].startswith('d'):
                print(f"Entering directory: {current_dir}/{item_name}")
                dfs.extend(traverse_ftp_directory(f"{current_dir}/{item_name}"))
                ftp.cwd('..')
            else:
                if item_name.endswith('.csv') and "playerpositioning" not in item_name.lower():
                    print(f"Found file: {current_dir}/{item_name}")

                    file_date = extract_date_from_filename(item_name)

                    if file_date:
                        if start_date and file_date < start_date:
                            print(f"Skipping file: {current_dir}/{item_name} (before {start_date.date()})")
                            continue
                        print(f"Downloading file: {current_dir}/{item_name}")
                        file_content = BytesIO()
                        
                        try:
                            ftp.retrbinary(f'RETR {current_dir}/{item_name}', file_content.write)
                            file_content.seek(0)
                            df = pd.read_csv(file_content, encoding='utf-8')
                            dfs.append(df)
                        except error_perm as e:
                            print(f"Permission error or file not found: {current_dir}/{item_name}. Skipping this file.")
                            continue
                        except Exception as e:
                            print(f"Error downloading {current_dir}/{item_name}: {e}")
                            continue
                    else:
                        print(f"No date found in filename: {current_dir}/{item_name}, skipping.")

        return dfs

    all_dfs = []

    for remote_dir in remote_dirs:
        print(f"Processing directory: {remote_dir}")
        all_dfs.extend(traverse_ftp_directory(remote_dir))

    concatenated_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    ftp.quit()

    return concatenated_df

yesterday = datetime.now() - timedelta(days=1)
yesterday_str = yesterday.strftime("%Y-%m-%d")
year = yesterday.strftime("%Y")
month = yesterday.strftime("%m")
day = yesterday.strftime("%d")


ftp_host = "ftp.trackmanbaseball.com"
ftp_user = "Georgia"
ftp_password = "UGAbaseball"
remote_dirs = [
    f"/practice/{year}/{month}/{day}/",
    f"/v3/{year}/{month}/{day}/csv/"
]

start_date = yesterday

print("data_etl.py ran successfully")