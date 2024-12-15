CREATE TABLE diigo AS SELECT * FROM '1722120_csv_2024_12_15_f3bd5.csv';

CREATE OR REPLACE TABLE raw_tags AS (SELECT UNNEST(tags) AS tag FROM (SELECT (string_split(tags, ',')) as tags from diigo where tags != 'no_tag'));

COPY diigo TO 'diigo.json';
