#
#  Copyright Alpinereplay Inc., 2026. All rights reserved.
#  Authors: Daniel Furman
#
"""
Retrieve facility stats from the Trace Vision API using the `facilityResult`
query.

This script queries the `facilityResult` endpoint, which returns the list of
facility stats active at the given point in time. Each `FacilityResult` has a
`start_time`, `end_time`, `stat_type`, and `stat_value`.

See: https://api.tracevision.com/graphql/v1/docs/queries/facilityResult

Usage:
1.  You will need a customer ID and API key to use this script. Contact us to
    get these. We will also share the API URL.
2.  You will need to know the facility ID and the time at which you want to
    retrieve facility stats.
3.  Run the script using the command line:
        python get_facility_result.py \\
            --customer_id 1234 \\
            --api_key "your_api_key" \\
            --api_url "api_url" \\
            --facility_id 5678 \\
            --time "2026-05-21T12:00:00Z"

    Optionally write the response to a JSON file:
        python get_facility_result.py ... --output facility_result.json
"""
import argparse
import json

import pandas as pd
import requests
from graphql_query import Argument, Field, Operation, Query, Variable


def build_facility_result_query():
    """
    Build the GraphQL operation string for the `facilityResult` query.

    :return query_string: Rendered GraphQL operation string
    :return variables_spec: Tuple of (token_var, facility_id_var, time_var) for
        reference (not strictly needed by the caller).
    """
    query_token = Variable(name="token", type="CustomerToken!")
    query_facility_id = Variable(name="facility_id", type="Int!")
    query_time = Variable(name="time", type="DateTime!")

    arg_token = Argument(name="token", value=query_token)
    arg_facility_id = Argument(name="facility_id", value=query_facility_id)
    arg_time = Argument(name="time", value=query_time)

    facility_result_query = Query(
        name="facilityResult",
        arguments=[arg_token, arg_facility_id, arg_time],
        fields=["start_time", "end_time", "stat_type", "stat_value"],
    )

    operation = Operation(
        type="query",
        name="facilityResult",
        variables=[query_token, query_facility_id, query_time],
        queries=[facility_result_query],
    )

    return operation.render()


def get_facility_result(customer_id, api_key, api_url, facility_id, time):
    """
    Query the `facilityResult` endpoint for stats active at a given time.

    :param customer_id: Customer (division) ID
    :param api_key: API key
    :param api_url: API URL
    :param facility_id: Facility ID to query
    :param time: ISO 8601 DateTime string (e.g. "2026-05-21T12:00:00Z")
    :return facility_results: List of dicts, one per FacilityResult, with keys
        `start_time`, `end_time`, `stat_type`, and `stat_value`
    """
    query_string = build_facility_result_query()
    variables = {
        "token": {"customer_id": customer_id, "token": api_key},
        "facility_id": facility_id,
        "time": time,
    }

    print(
        f"Querying facilityResult for facility_id={facility_id}, time={time}, "
        f"customer_id={customer_id}"
    )
    response = requests.post(
        api_url,
        json={"query": query_string, "variables": variables},
    )

    if response.status_code != 200:
        raise ValueError(
            f"Received status code {response.status_code} from API. "
            f"Body: {response.text}"
        )

    response_json = response.json()
    if response_json is None:
        raise ValueError(
            "Received null response from the API. Check the API URL, customer "
            "ID, and API key."
        )
    if "errors" in response_json:
        print("Errors received from the API:")
        for cur_error in response_json["errors"]:
            print(cur_error.get("message", cur_error))

    facility_results = response_json.get("data", {}).get("facilityResult", [])
    print(f"Retrieved {len(facility_results)} facility result entries.")
    return facility_results


def main():
    """
    Parse command line arguments and retrieve facility stats.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--customer_id", required=True, type=int, help="Customer (division) ID"
    )
    ap.add_argument("--api_key", required=True, type=str, help="API key")
    ap.add_argument("--api_url", required=True, type=str, help="API URL")
    ap.add_argument(
        "--facility_id", required=True, type=int, help="Facility ID"
    )
    ap.add_argument(
        "--time",
        required=True,
        type=str,
        help=(
            "Point in time to retrieve facility stats. ISO 8601 DateTime "
            'format, e.g. "2026-05-21T12:00:00Z"'
        ),
    )
    ap.add_argument(
        "--output",
        required=False,
        type=str,
        default=None,
        help="Optional path to write the JSON response to.",
    )
    args = vars(ap.parse_args())

    facility_results = get_facility_result(
        customer_id=args["customer_id"],
        api_key=args["api_key"],
        api_url=args["api_url"],
        facility_id=args["facility_id"],
        time=args["time"],
    )

    if not facility_results:
        print("No facility result entries returned.")
        return

    # Print a tabular view of the results.
    results_df = pd.DataFrame(facility_results)
    print("\nFacility results:")
    print(results_df.to_string(index=False))

    if args["output"]:
        with open(args["output"], "w") as f:
            json.dump(facility_results, f, indent=4)
        print(f"\nWrote response to {args['output']}")


if __name__ == "__main__":
    main()
