import os


class GCS:
    """
    A class to interact with Google Cloud Storage (GCS).

    Attributes:
        storage_client (google.cloud.storage.Client): A client for interacting with GCS.
    """

    def __init__(self, storage_client):
        """
        Initializes the GCS class with a storage client.

        Args:
            storage_client (google.cloud.storage.Client): The GCS client to use for operations.
        """
        self.storage_client = storage_client

    def get_bucket(self, bucket_name):
        """
        Retrieves a bucket from GCS by its name.

        Args:
            bucket_name (str): The name of the bucket.

        Returns:
            google.cloud.storage.bucket.Bucket: The GCS bucket.
        """
        return self.storage_client.bucket(bucket_name)

    def list_buckets(self):
        """
        Lists all buckets available in the GCS project.

        Returns:
            list: A list of bucket names.
        """
        buckets = self.storage_client.list_buckets()
        return [bucket.name for bucket in buckets]

    def list_blobs(self, bucket_name):
        """
        Lists all blobs in a given bucket.

        Args:
            bucket_name (str): The name of the bucket.

        Returns:
            list: A list of blob names.
        """
        bucket = self.get_bucket(bucket_name)
        blobs = bucket.list_blobs()
        return [blob.name for blob in blobs]

    def download_blob(self, bucket_name, source_blob_name, destination_file_name):
        """
        Downloads a blob from GCS to a local file.

        Args:
            bucket_name (str): The name of the bucket containing the blob.
            source_blob_name (str): The name of the blob to download.
            destination_file_name (str): The local file path where the blob should be saved.
        """
        bucket = self.get_bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        try:
            blob.download_to_filename(destination_file_name)
            print(
                f"Downloaded {source_blob_name} from bucket {bucket_name} to {destination_file_name}."
            )
        except Exception as e:
            print(
                f"Error downloading {source_blob_name} from {bucket_name} to {destination_file_name}: {e}"
            )
            if os.path.exists(destination_file_name):
                os.remove(destination_file_name)
