import os
import shutil
import logging

def delete_files_in_directory(directory):
    """Delete all files in the specified directory."""
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logging.error(f"Failed to delete {file_path}. Reason: {e}")

def cleanup():
    logging.basicConfig(filename='cleanup.log', level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

    # Directories and files to clean
    tensorboard_logs_dir = './tensorboard_logs'
    screenshots_dir = r'C:\Users\marvi\Pinball Wizard\Screenshots'
    final_model_zip = 'final_model.zip'
    interrupted_model_zip = 'interrupted_model.zip'

    # Delete tensorboard logs
    logging.info(f"Deleting files in {tensorboard_logs_dir}")
    delete_files_in_directory(tensorboard_logs_dir)

    # Delete screenshots
    logging.info(f"Deleting files in {screenshots_dir}")
    delete_files_in_directory(screenshots_dir)

    # Delete model zip files
    for model_zip in [final_model_zip, interrupted_model_zip]:
        if os.path.exists(model_zip):
            try:
                os.remove(model_zip)
                logging.info(f"Deleted {model_zip}")
            except Exception as e:
                logging.error(f"Failed to delete {model_zip}. Reason: {e}")

if __name__ == "__main__":
    cleanup()
