-- ranks country origins of bands, ordered by the number of (non-unique) fans
-- metal_bands(id, band_name, fans, formed, origin, split, style)

SELECT origin, SUM(fans) AS nb_fans from metal_bands
GROUP BY origin ORDER BY nb_fans DESC;