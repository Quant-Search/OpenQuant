//+------------------------------------------------------------------+
//|                                          OpenQuantVisualizer.mq5 |
//|                                  Copyright 2025, OpenQuant Robot |
//|                                             https://openquant.io |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, OpenQuant Robot"
#property link      "https://openquant.io"
#property version   "1.00"
#property description "Visualizes OpenQuant trading signals from CSV"

// Inputs
input string InpFileName = "signals.csv"; // CSV File Name (in MQL5/Files)
input int    InpTimer    = 1;             // Timer Interval (seconds)

// Global variables
datetime last_file_time = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   // Set timer
   EventSetTimer(InpTimer);
   Print("OpenQuant Visualizer Started. Monitoring: ", InpFileName);
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
   ObjectsDeleteAll(0, "OQ_");
  }
//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
  {
   // Check if file exists
   if(!FileIsExist(InpFileName))
      return;

   // Open file
   int handle = FileOpen(InpFileName, FILE_READ|FILE_CSV|FILE_ANSI, ",");
   if(handle == INVALID_HANDLE)
     {
      Print("Failed to open file: ", InpFileName);
      return;
     }

   // Read header
   if(!FileIsEnding(handle))
      FileReadString(handle); // Skip header line

   // Read rows
   while(!FileIsEnding(handle))
     {
      string line = FileReadString(handle); // Read whole line if possible, or fields
      // CSV format: Symbol,Side,Weight,Timestamp
      // But FileReadString with CSV delimiter reads one field at a time
      
      // Let's re-open without CSV flag to read lines manually for robustness, 
      // or just assume strict structure. Let's stick to standard CSV read.
     }
   FileClose(handle);
   
   // Re-implementing read logic properly
   ReadAndDraw();
  }

void ReadAndDraw()
{
   int handle = FileOpen(InpFileName, FILE_READ|FILE_CSV|FILE_ANSI, ",");
   if(handle == INVALID_HANDLE) return;

   // Skip Header: Symbol,Side,Weight,Timestamp
   string h1=FileReadString(handle); 
   string h2=FileReadString(handle);
   string h3=FileReadString(handle);
   string h4=FileReadString(handle);

   while(!FileIsEnding(handle))
   {
      string sym = FileReadString(handle);
      string side = FileReadString(handle);
      string weight_str = FileReadString(handle);
      string time_str = FileReadString(handle);
      
      if(sym == "" || time_str == "") continue;
      
      // Filter for current chart symbol
      if(sym != _Symbol) continue;
      
      // Parse Time (Format expected: YYYY-MM-DD HH:MM:SS)
      datetime time = StringToTime(time_str);
      
      // Create Object Name
      string obj_name = "OQ_Arrow_" + time_str;
      
      if(ObjectFind(0, obj_name) < 0)
      {
         // Determine Type and Price
         ENUM_OBJECT obj_type;
         double price = 0;
         int arrow_code = 0;
         color clr = clrNONE;
         
         // Get bar data for price
         MqlRates rates[];
         int copied = CopyRates(sym, PERIOD_CURRENT, time, 1, rates);
         if(copied > 0)
         {
            if(StringFind(side, "BUY") >= 0)
            {
               price = rates[0].low;
               arrow_code = 233; // Up Arrow
               clr = clrGreen;
            }
            else if(StringFind(side, "SELL") >= 0)
            {
               price = rates[0].high;
               arrow_code = 234; // Down Arrow
               clr = clrRed;
            }
            
            if(price > 0)
            {
               ObjectCreate(0, obj_name, OBJ_ARROW, 0, time, price);
               ObjectSetInteger(0, obj_name, OBJPROP_ARROWCODE, arrow_code);
               ObjectSetInteger(0, obj_name, OBJPROP_COLOR, clr);
               ObjectSetString(0, obj_name, OBJPROP_TEXT, side + " " + weight_str);
               ObjectSetInteger(0, obj_name, OBJPROP_WIDTH, 2);
               Print("Drew signal: ", side, " at ", time_str);
            }
         }
      }
   }
   FileClose(handle);
}
//+------------------------------------------------------------------+
