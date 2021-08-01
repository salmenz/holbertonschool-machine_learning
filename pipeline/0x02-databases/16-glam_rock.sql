-- lists all bands with Glam rock as their main style, ranked by their longevity
-- metal_bands(id, band_name, fans, formed, origin, split, style)

SELECT band_name, IF(split IS NULL,
    (YEAR(NOW()) - formed),
    split - formed) AS lifespan
    FROM metal_bands WHERE style LIKE '%Glam Rock%'
    ORDER BY lifespan DESC;