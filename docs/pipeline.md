# Pipeline Documentation

## Final Workflow

```text
Image
  -> YOLO-World Detection
  -> Crop OCR
  -> Brand Mapping
  -> Price OCR
  -> Bbox-area Shelf-Space Estimation
  -> Product-Price Linking
  -> Final JSON + Annotated Image + Streamlit
```

## Step 1: YOLO-World Detection

`detector.py` loads the existing YOLO-World model and detects product-like
objects. This logic is unchanged and remains responsible only for product
bounding boxes.

## Step 2: Crop OCR

`ocr_service.py` uses EasyOCR on each product crop. The OCR text is kept in JSON
and passed to brand mapping.

## Step 3: Brand Mapping

`brand_mapper.py` maps OCR words to clean product or brand names such as
`Coca-Cola`, `Pepsi`, `Fanta`, or `Other`.

## Step 4: Price OCR

`price_ocr.py` runs full-image EasyOCR and extracts visible shelf prices. It
supports common formats such as:

- `Rs 20`
- `INR 20`
- `20/-`
- `20`
- `50`
- `99`
- `125`

The price tag bounding box is kept close to the actual price text so the final
annotation does not draw huge row-level boxes.

## Step 5: Product-Price Linking

`price_linker.py` links products to prices with two rules:

1. Text match: product label is compared with price tag text.
2. Nearest-below fallback: used only when text matching fails.

## Step 6: Shelf-Space Estimation

`metrics.py` estimates brand-wise shelf-space percentage.

Shelf-space estimation is calculated using product bounding-box area as a
lightweight product-region segmentation approximation.

```text
brand_area = sum(product bbox area for that brand)
shelf_space_percent = brand_area / total_product_bbox_area * 100
```

This is an explicit project assumption for a lightweight assignment pipeline:
the detected product bounding box is treated as the practical product region for
area comparison.

## Step 7: Outputs

The project saves one final image:

```text
outputs/<image_name>_detected.jpg
```

The image contains:

- green product boxes,
- clean product/brand labels,
- orange price tag boxes,
- clean price labels only,
- no segmentation mask overlay.

The JSON contains:

- `image_name`
- `total_products`
- `brands`
- `shelf_space_percent`
- `ocr_labels`
- `detections`

Each detection includes `area_source: "bbox"` to make the shelf-space area
method explicit.

## Step 8: Streamlit

`app.py` provides a lightweight GUI for:

- uploading an image,
- viewing original and annotated images,
- reviewing product counts,
- viewing brand-wise counts,
- viewing shelf-space percentages,
- viewing compact product cards,
- viewing OCR labels / price tags,
- viewing product rows,
- previewing and downloading JSON.

## Design Notes

- YOLO-World detection remains the primary product detector.
- EasyOCR remains the default OCR engine for product labels and prices.
- The final annotated image is intentionally clean and contains only boxes and labels.
- The implementation intentionally stays simple for final assignment submission.
