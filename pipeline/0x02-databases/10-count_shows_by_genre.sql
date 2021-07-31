-- lists all genres from hbtn_0d_tvshows and displays the number of shows linked to each

SELECT tv_genres.name AS genre, COUNT(tv_show_genres.genre_id)
    AS number_of_shows FROM tv_show_genres
    JOIN tv_genres ON tv_show_genres.genre_id = tv_genres.id
    GROUP by tv_genres.name
    ORDER BY number_of_shows DESC;