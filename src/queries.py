"""Sanitised warehouse-query templates used by the app and scheduled job.

Table names are intentionally generic. A deployment supplies the catalog and
schema through environment variables while preserving the query shape,
deduplication rules, and output contract used by the pipeline.
"""

import os


CATALOG = os.environ.get("WAREHOUSE_CATALOG", "analytics")
SCHEMA = os.environ.get("WAREHOUSE_SCHEMA", "aviation")


def _table(name: str) -> str:
    return f"{CATALOG}.{SCHEMA}.{name}"


def ground_events_query(year: int, month: int, day: int) -> str:
    """Fetch active maintenance and standby events for one operating day."""
    date_str = f"{year}-{month:02d}-{day:02d}"
    return f"""
    SELECT aircraft_registration AS tailnumber, event_code AS checktypecode,
           airport_code AS base, operator_code AS aoc
    FROM {_table('ground_activity_details')}
    WHERE status = 'ACTIVE'
      AND CAST(schedule_start AS DATE) <= '{date_str}'
      AND CAST(schedule_end AS DATE) >= '{date_str}'
    QUALIFY ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY updated_at DESC) = 1
    """


def daily_input_query(year: int, month: int, day: int) -> str:
    """Build the forecast-window input query expected by Stage 1.

    The result deliberately follows the production column contract while using
    neutral warehouse object names.
    """
    date_str = f"{year}-{month:02d}-{day:02d}"
    return f"""
    WITH schedule AS (
        SELECT
            ROW_NUMBER() OVER (PARTITION BY flight_number, DATE(scheduled_departure)
                               ORDER BY updated_at DESC) AS row_num,
            flight_number AS flightnumber,
            aircraft_registration AS tailnumber,
            scheduled_departure AS scheduledeparturedatetime,
            scheduled_arrival AS schedulearrivaldatetime,
            departure_iata AS departureairportcode,
            arrival_iata AS arrivalairportcode,
            flight_status AS flightstatuscode
        FROM {_table('flight_schedule')}
        WHERE scheduled_departure >= TIMESTAMP('{date_str}') + INTERVAL 3 HOURS
          AND scheduled_departure <  TIMESTAMP('{date_str}') + INTERVAL 27 HOURS
          AND flight_status != 'CANCELLED'
    ),
    latest_aircraft_stats AS (
        SELECT aircraft_registration, max_takeoff_weight AS acft_mtow,
               performance_type AS perf_type, performance_correction AS perf_corr
        FROM {_table('aircraft_performance_latest')}
    ),
    latest_capacity AS (
        SELECT aircraft_registration, seat_capacity AS aircraftcapacity
        FROM {_table('aircraft_capacity_latest')}
    ),
    route_distance AS (
        SELECT departure_iata, arrival_iata,
               AVG(airway_distance_km) AS awy_dist_km
        FROM {_table('route_distance_history')}
        WHERE airway_distance_km IS NOT NULL
        GROUP BY departure_iata, arrival_iata
    )
    SELECT s.*, a.acft_mtow, a.perf_type, a.perf_corr, c.aircraftcapacity,
           d.awy_dist_km
    FROM schedule s
    LEFT JOIN latest_aircraft_stats a ON s.tailnumber = a.aircraft_registration
    LEFT JOIN latest_capacity c ON s.tailnumber = c.aircraft_registration
    LEFT JOIN route_distance d ON s.departureairportcode = d.departure_iata
                             AND s.arrivalairportcode = d.arrival_iata
    WHERE s.row_num = 1
    """
