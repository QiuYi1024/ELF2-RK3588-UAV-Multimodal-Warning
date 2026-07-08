#!/usr/bin/env python3

import unittest

from rid900_reader import parse_line


GBRID = (
    "$GBRID,-38,43,DEMO_SERIAL_001,DJI,Matrice 4E,2,1,2,"
    "0.000000,0.000000,0.5,100.0,100.5,181,0,0,11,3,"
    "0.000000,0.000000,0.0,1*189"
)
HBRID = "$HBRID,heartbeat,DEMO_DEVICE_001,-81,10,0.000000,0.000000,16.6,1*347"
HBRID_NO_FIX = "$HBRID,heartbeat,DEMO_DEVICE_002,-81,0,,,,2*14"


class Rid900ParserTest(unittest.TestCase):
    def test_gbrid_sample(self):
        result = parse_line(GBRID)
        self.assertEqual(result["type"], "rid900_target")
        self.assertEqual(result["serial_number"], "DEMO_SERIAL_001")
        self.assertEqual(result["vendor"], "DJI")
        self.assertEqual(result["product_type"], "Matrice 4E")
        self.assertAlmostEqual(result["drone_longitude"], 0.000000)
        self.assertAlmostEqual(result["pilot_latitude"], 0.000000)
        self.assertEqual(result["checksum_text"], "189")

    def test_hbrid_sample(self):
        result = parse_line(HBRID)
        self.assertEqual(result["type"], "rid900_heartbeat")
        self.assertEqual(result["device_id"], "DEMO_DEVICE_001")
        self.assertAlmostEqual(result["module_latitude"], 0.000000)
        self.assertEqual(result["fix_status"], 1)

    def test_hbrid_without_gps_fix(self):
        result = parse_line(HBRID_NO_FIX)
        self.assertEqual(result["device_id"], "DEMO_DEVICE_002")
        self.assertEqual(result["module_longitude"], 0.0)
        self.assertEqual(result["module_latitude"], 0.0)
        self.assertEqual(result["fix_status"], 2)
        self.assertEqual(result["checksum_text"], "14")

    def test_rejects_unknown_sentence(self):
        with self.assertRaises(ValueError):
            parse_line("$GPGGA,invalid")


if __name__ == "__main__":
    unittest.main()
