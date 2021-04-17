DATASET_NAME = "product-image-mturk-30k"

S3_BUCKET = "data-science-product-image"
S3_PATH = "s3://" + S3_BUCKET + "/datasets/" + DATASET_NAME
MANIFEST_FILE_NAME = "input.manifest"

IMAGES_PATH = S3_PATH + "/images"
LABEL_JOB_PATH = S3_PATH + "/labeling"
MANIFEST_FILE_PATH = LABEL_JOB_PATH + "/" + MANIFEST_FILE_NAME

# max number of products to add to the manifest
MAX_MANIFEST_SIZE = 51000

# product collation stuff
PRODUCTS_URI = "http://prod-rs-product-service.rslocal/v1/retailer_product_references"

PRODUCTS_KEY = "retailer_products"
PRODUCT_KEYS = ["id", "title", "description", "usd"]

PRODUCTS_FILENAME = "products.csv"
PRODUCTS_PATH = S3_PATH + "/collation/" + PRODUCTS_FILENAME

PRODUCTS_COLLATED_FILENAME = "products-collated.json"
PRODUCTS_COLLATED_PATH = S3_PATH + "/collation/" + PRODUCTS_COLLATED_FILENAME