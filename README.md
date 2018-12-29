curl -u vanbelle:summertime1  -H 'Content-Type: application/json' --data-binary '{
  "timeSpentSeconds": 300,
  "dateStarted": "2018-09-21T00:00:00.000",
  "author": {
    "name": "vanbelle"
  },
  "comment": "testing",
  "issue": {
    "key": "SRCHRD-392"
  },
  "workAttributeValues": [
    {
      "value": "myattributetest",
      "workAttribute": {
        "type": {
          "value": "INPUT_FIELD"
        },
        "name": "myattributetestname"
      },
      "name": "attribute1"
    }
  ]
}' 'https://jira.textkernel.nl/rest/tempo-timesheets/3/worklogs'
