import numpy as np
import datetime
import pandas as pd
import pandas_market_calendars as mcal
import pytz

import misc_functions as misc_fns


class listing:
    """
    """

    def __init__(self, ticker, v_name=None, exchange=None):
        """
        """
        self.ticker = ticker
        self.v_name = v_name
        self.exchange = exchange
        self.calender = mcal.get_calendar(self.exchange)
        
        self.file = "../data/{}.csv".format(self.ticker)
        self.data = pd.read_csv(self.file,
                                sep=',',
                                index_col="Date"
                                )
        self.remove_non_trading_days()

        self.adj_close = self.data.loc[:,"Adj Close"]
        self.log_returns = np.log(self.adj_close).diff().dropna()

    

    def get_schedule(self, start_date=None, end_date=None):


        if start_date is None:
            start_date = self.data.index[0]
        if end_date is None:
            end_date = "2099-12-31"

        schedule = self.calender.schedule(start_date=start_date, end_date=end_date)

        false_list = {i: {"positive": None, "negative": None} for i in mcal.get_calendar_names()}
        false_date = false_list.copy()

        false_list["LSE"]["positive"] = [
                                            "1995-05-08", "1999-12-31", 
                                            "1977-06-07", "2002-06-03", "2002-06-04", 
                                            "1973-11-14", "1973-11-14","2011-04-29", 
                                            "1999-12-31"
                                        ]
        false_list["LSE"]["negative"] = [
                                            "1995-05-01", "2020-05-04", 
                                            "2002-05-27", "2012-05-28"
                                        ]

        for i in false_list:
            for j in false_list[i]:
                if false_list[i][j] is not None:
                    false_date[i][j] = np.array([np.datetime64(k) for k in false_list[i][j]])
        
        schedule = schedule.iloc[np.logical_not(np.isin(schedule.index, false_date[self.exchange]["positive"]))]

        if false_date[self.exchange]["negative"] is not None:

            for k in false_date[self.exchange]["negative"]:
                if np.isin(k, schedule.index) == False:
                    new_date = pd.Timestamp(k)

                    past = schedule.index < new_date
                    ref = np.sum(past)
                    if np.all(past):
                        ref = np.size(past) - 1                    

                    new_pd_data = schedule.iloc[ref] - pytz.utc.localize(schedule.index[ref])
                    new_pd = pd.DataFrame([new_pd_data + new_date], index=[new_date], dtype="datetime64[ns, UTC]")
                    schedule = schedule.append(new_pd)

            schedule = schedule.sort_index()
        
        return schedule



    def remove_non_trading_days(self):

        self.data.index = np.array([np.datetime64(self.data.index[k]) for k in range(np.size(self.data.index))])
        
        self.schedule = self.get_schedule()

        sch_dates = self.schedule.index[self.schedule.index <= self.data.index[-1]]

        self.data = self.data.iloc[np.where(np.isin(self.data.index, sch_dates))[0]]




    def find_holidays(self, Y_end=None, Y_start=None):
        """
        """
        hdays = holidays(Y_end=Y_end, Y_start=Y_start)
        return hdays.Country[self.Country]()

    def set_busdaycal(self):
        """
        """
        holidays = self.find_holidays()

        all_days = np.arange(self.data.index[0], self.data.index[-1], step=np.timedelta64(1, 'D'), dtype='datetime64[D]')
        bus_days = all_days[np.is_busday(all_days,  holidays=holidays)]
        close_days = close_days = np.sort(np.concatenate([holidays, bus_days[np.logical_not(np.isin(bus_days, np.array(self.data.index, dtype="datetime64[D]")))]]))

        self.busdaycal = np.busdaycalendar(holidays=close_days)





class holidays:
    """Class for finding holidays by Country
    """

    def __init__(self, Y_end=None, Y_start=None):
        """
        """
        if Y_end is None:
            self._Y_end = 2100
        else:
            self._Y_end = Y_end
        if Y_start is None:
            self._Y_start = 1871
        else:
            self._Y_end = Y_end

        self.Country =  {
                            "England": self.England_holidays,
                            "US": self.US_holidays,
                        }

    

    def Y_range(self, Y=None, Y_start=None, Y_end=None, hard_start=False, hard_end=False):

        if Y_start is None:
            Y_start = self._Y_start
        if Y_end is None:
            Y_end = self._Y_end
        if Y is None:
            Y = np.array(range(Y_start, Y_end))
        else:
            Y = misc_fns.make_iterable_array(Y)
            if hard_start:
                Y = Y[Y >= Y_start]
            if hard_end:
                Y = Y[Y <= Y_end]

        return Y, Y_start, Y_end




    def fixed_date(self, day, month, Y=None, Y_start=None, Y_end=None, hard_start=None, hard_end=None):

        [Y, Y_start, Y_end] = self.Y_range(Y=Y, Y_start=Y_start, Y_end=Y_end, hard_start=hard_start, hard_end=hard_end)

        if Y.size > 0:
            fixed_date = np.array([np.datetime64(datetime.datetime(year=Y[i], month=month, day=day).strftime("%Y-%m-%d")) for i in range(np.size(Y))])

            return fixed_date


    def day_of_week_month(self, weekmask, month, index=0, Y=None, Y_start=None, Y_end=None, hard_start=None, hard_end=None):

        [Y, Y_start, Y_end] = self.Y_range(Y=Y, Y_start=Y_start, Y_end=Y_end, hard_start=hard_start, hard_end=hard_end)

        if Y.size > 0:
            month_64 = np.array([np.datetime64(datetime.datetime(year=Y[k], month=month, day=1).strftime("%Y-%m")) for k in range(np.size(Y))])
            if index < 0:
                month_64 += np.timedelta64(1, 'M')

            return np.busday_offset(month_64, index, roll='forward', weekmask=weekmask)


    def OBS_closest_weekday(self, dates):

        dates_OBS_fri = dates[np.is_busday(dates, weekmask="Sat")] - np.timedelta64(1, 'D')
        dates_OBS_mon = dates[np.is_busday(dates, weekmask="Sun")] + np.timedelta64(1, 'D')
        
        return np.sort(np.concatenate((dates_OBS_fri, dates_OBS_mon)))



    def OBS_closest_weekday_shift(self, dates):

        dates[np.is_busday(dates, weekmask="Sat")] -= np.timedelta64(1, 'D')
        dates[np.is_busday(dates, weekmask="Sun")] += np.timedelta64(1, 'D')




    def Meeus_Gregorian(self, Y=None, Y_start=None, Y_end=None, hard_start=False, hard_end=False):
        """ Returns date of Easter Sunday (Western) in year Y in the Gregorian calender
        """

        [Y, Y_start, Y_end] = self.Y_range(Y=Y, Y_start=Y_start, Y_end=Y_end, hard_start=hard_start, hard_end=hard_end)
        Y = np.array(Y)

        a = np.mod(Y,19)
        b = np.array(Y/100, dtype=int)
        c = np.mod(Y,100)
        d = np.array(b/4, dtype=int)
        e = np.mod(b,4)
        f = np.array((b+8)/25, dtype=int)
        g = np.array((b-f+1)/3, dtype=int)
        h = np.mod((19*a + b - d - g + 15), 30)
        i = np.array(c/4, dtype=int)
        k = np.mod(c,4)
        l = np.mod((32 + 2*e + 2*i - h - k), 7)
        m = np.array((a + 11*h + 22*l)/451, dtype=int)
    
        month = np.array((h + l - 7*m + 114)/31, dtype=int)
        day = np.mod((h + l - 7*m + 114), 31) + 1

        dates = np.array([np.datetime64(datetime.datetime(year=Y[i], month=month[i], day=day[i]).strftime("%Y-%m-%d")) for i in range(np.size(Y))])

        if dates.size > 0:
            return dates




    def NYD(self, Y=None, Y_start=None, Y_end=None, hard_start=False, hard_end=False):

        return self.fixed_date(day=1, month=1, Y=Y, Y_start=Y_start, Y_end=Y_end, hard_start=hard_start, hard_end=hard_end)


    def Christmas_Day(self, Y=None, Y_start=None, Y_end=None, hard_start=False, hard_end=False):

        return self.fixed_date(day=25, month=12, Y=Y, Y_start=Y_start, Y_end=Y_end, hard_start=hard_start, hard_end=hard_end)


    def Easter_Sunday_West(self, Y=None, Y_start=None, Y_end=None, hard_start=False, hard_end=False):

        return self.Meeus_Gregorian(Y=Y, Y_start=Y_start, Y_end=Y_end, hard_start=hard_start, hard_end=hard_end)

    def Good_Friday_West(self, Y=None, Y_start=None, Y_end=None, hard_start=False, hard_end=False):

        return np.busday_offset(self.Meeus_Gregorian(Y=Y, Y_start=Y_start, Y_end=Y_end, hard_start=hard_start, hard_end=hard_end), -1, roll="forward", weekmask="Fri")

    def Easter_Monday_West(self, Y=None, Y_start=None, Y_end=None, hard_start=False, hard_end=False):

        return np.busday_offset(self.Meeus_Gregorian(Y=Y, Y_start=Y_start, Y_end=Y_end, hard_start=hard_start, hard_end=hard_end), 0, roll="forward", weekmask="Mon")




#---#=============================================
    # England Holidays (LSE)
    #=============================================

    def England_holidays(self, Y=None):

            England_holiday_list =  [   
                                        self.England_NYD_OBS(Y=Y),
                                        self.England_Good_Friday(Y=Y),
                                        self.England_Easter_Monday(Y=Y),
                                        self.England_May_Bank(Y=Y),
                                        self.England_Spring_Bank(Y=Y),
                                        self.England_Summer_Bank(Y=Y),
                                        self.England_Christmas_Day_OBS(Y=Y),
                                        self.England_Boxing_Day_OBS(Y=Y),
                                        self.England_Unique(Y=Y)
                                    ]
            

            is_none = np.where([elem is None for elem in England_holiday_list])[0]
            if is_none.size != 0:
                for i in range(is_none.size):
                    England_holiday_list.pop(is_none[i])

            if len(England_holiday_list) > 0:
                return np.sort(np.concatenate(England_holiday_list))
            


    def England_NYD_OBS(self, Y=None):

        NYD = self.NYD(Y=Y, Y_start=1974, hard_start=True)
        if NYD is not None:
            return np.busday_offset(NYD, 0, roll="forward")


    def England_Good_Friday(self, Y=None):

        Easter_Sunday = self.Easter_Sunday_West(Y=Y, Y_start=1834, hard_start=True)
        if Easter_Sunday is not None:
            return np.busday_offset(Easter_Sunday, -1, roll="forward", weekmask="Fri")


    def England_Easter_Monday(self, Y=None):

        Easter_Sunday = self.Easter_Sunday_West(Y=Y, Y_start=1871, hard_start=True)
        if Easter_Sunday is not None:
            return np.busday_offset(Easter_Sunday, 0, roll="forward", weekmask="Mon")


    def England_May_Bank(self, Y=None):

        Current = self.Y_range(Y=Y, Y_start=1971, hard_start=True)

        England_May_Bank = self.day_of_week_month(weekmask="Mon", month=5, index=0, Y=Current[0], Y_start=Current[1], hard_start=True)

        if England_May_Bank is None:
            return
        else:
            # VE Day anniversary
            England_May_Bank = misc_fns.make_iterable_array(England_May_Bank)
            if np.any(Current[0] == 1995):
                England_May_Bank[np.where(Current[0]==1995)[0]] = np.datetime64("1995-05-08")
            if np.any(Current[0] == 2020):
                England_May_Bank[np.where(Current[0]==2020)[0]] = np.datetime64("2020-05-08")

            return England_May_Bank

        
    def England_Spring_Bank(self, Y=None):


        Whilt = self.Y_range(Y=Y, Y_start=1871, Y_end=1977, hard_start=True, hard_end=True)
        post_Whilt = self.Y_range(Y=Y, Y_start=1978, hard_start=True)

        England_Spring_Bank = []
        if np.size(Whilt[0]) > 0:
            England_Spring_Bank.append(self.Easter_Sunday_West(Y=Whilt[0]) + np.timedelta64(50, 'D'))
        if np.size(post_Whilt[0]) > 0:
            England_Spring_Bank_post_Whilt = misc_fns.make_iterable_array(self.day_of_week_month(weekmask="Mon", month=5, index=-1, Y=post_Whilt[0], Y_start=post_Whilt[1]))

            # QEII Jubilee
            if np.any(post_Whilt[0] == 2002):
                England_Spring_Bank_post_Whilt[np.where(post_Whilt[0] == 2002)[0]] = np.datetime64("2002-06-03")
            if np.any(post_Whilt[0] == 2012):
                England_Spring_Bank_post_Whilt[np.where(post_Whilt[0] == 2012)[0]] = np.datetime64("2012-06-05")

            England_Spring_Bank.append(England_Spring_Bank_post_Whilt)

        if len(England_Spring_Bank) > 0:
            return np.sort(np.concatenate(England_Spring_Bank))

    def England_Summer_Bank(self, Y=None):

        Old = self.Y_range(Y=Y, Y_start=1871, Y_end=1964, hard_start=True, hard_end=True)
        post_Old = self.Y_range(Y=Y, Y_start=1965, hard_start=True)

        England_Summer_Bank = []
        if np.size(Old[0]) > 0:
            England_Summer_Bank.append(self.day_of_week_month(weekmask="Mon", month=8, index=1, Y=Old[0], Y_start=Old[1], Y_end=Old[2]))
        if np.size(post_Old[0]) > 0:
            England_Summer_Bank.append(self.day_of_week_month(weekmask="Mon", month=8, index=-1, Y=post_Old[0], Y_start=post_Old[1], Y_end=post_Old[2]))

        if len(England_Summer_Bank) > 0:
            return np.sort(np.concatenate(England_Summer_Bank))


    def England_Christmas_Day_OBS(self, Y=None):

        Christmas = self.Christmas_Day(Y=Y, Y_start=1834, hard_start=True)
        if Christmas is not None:
            return np.busday_offset(Christmas, 0, roll="forward")


    def England_Boxing_Day_OBS(self, Y=None):

        Christmas = self.Christmas_Day(Y=Y, Y_start=1871, hard_start=True)
        if Christmas is not None:
            return np.busday_offset(Christmas, 1, roll="forward")


    def England_Unique(self, Y=None):

        Unique = np.array(  [
                                np.datetime64("1973-11-14"),
                                np.datetime64("1977-06-07"),
                                np.datetime64("1981-07-29"),
                                np.datetime64("1999-12-31"),
                                np.datetime64("2011-04-29"),
                            ])

        [Y, Y_start, Y_end] = self.Y_range(Y=Y)

        years = np.array([np.datetime64("{}".format(Y[k])) for k in range(Y.size)], dtype='datetime64[Y]')
        Unique_years = np.array(Unique, dtype='datetime64[Y]')
        included = np.isin(Unique_years, years)

        if np.any(included):
            return Unique[included]




#---#=============================================
    # US Holidays (NYSE, NASDAQ)
    #=============================================

    def US_holidays(self, Y=None):

            US_holiday_list =   [   
                                    self.US_NYD_OBS(Y=Y),
                                    self.US_MLK_Day(Y=Y),
                                    self.US_Presidents_Day(Y=Y),
                                    self.US_Good_Friday(Y=Y),
                                    self.US_Memorial_Day(Y=Y),
                                    self.US_Independence_Day_OBS(Y=Y),
                                    self.US_Labor_Day(Y=Y),
                                    self.US_Thanksgiving(Y=Y),
                                    self.US_Christmas_Day_OBS(Y=Y),
                                ]
            
            is_none = np.where([elem is None for elem in US_holiday_list])[0]
            if is_none.size != 0:
                for i in range(is_none.size):
                    US_holiday_list.pop(is_none[i])

            if len(US_holiday_list) > 0:
                return np.sort(np.concatenate(US_holiday_list))


    def US_NYD_OBS(self, Y=None):

        US_NYD_OBS = self.NYD(Y=Y, Y_start=1885, hard_start=True)
        if US_NYD_OBS is not None:
            US_NYD_OBS[np.is_busday(US_NYD_OBS, weekmask="Sun")] += np.timedelta64(1, 'D')
            return US_NYD_OBS


    def US_MLK_Day(self, Y=None):

        return self.day_of_week_month(weekmask="Mon", month=1, index=2, Y=Y, Y_start=1998, hard_start=True)

    
    def US_Presidents_Day(self, Y=None):

        Old = self.Y_range(Y=Y, Y_start=1885, Y_end=1970, hard_start=True, hard_end=True)
        post_Old = self.Y_range(Y=Y, Y_start=1971, hard_start=True)

        US_Presidents_Day = []
        if np.size(Old[0]) > 0:
            US_Presidents_Day.append(self.fixed_date(day=22, month=2, Y=Old[0], Y_start=Old[1], Y_end=Old[2]))
        if np.size(post_Old[0]) > 0:
            US_Presidents_Day.append(self.day_of_week_month(weekmask="Mon", month=2, index=2, Y=post_Old[0], Y_start=post_Old[1], Y_end=post_Old[2]))

        if len(US_Presidents_Day) > 0:
            return np.sort(np.concatenate(US_Presidents_Day))


    def US_Good_Friday(self, Y=None):

        Easter_Sunday = self.Easter_Sunday_West(Y=Y, Y_start=1885, hard_start=True)
        if Easter_Sunday is not None:
            return np.busday_offset(Easter_Sunday, -1, roll="forward", weekmask="Fri")


    def US_Memorial_Day(self, Y=None):

        Old = self.Y_range(Y=Y, Y_start=1885, Y_end=1970, hard_start=True, hard_end=True)
        post_Old = self.Y_range(Y=Y, Y_start=1971, hard_start=True)

        US_Memorial_Day = []
        if np.size(Old[0]) > 0:
            US_Memorial_Day.append(self.fixed_date(day=30, month=5, Y=Old[0], Y_start=Old[1], Y_end=Old[2]))
        if np.size(post_Old[0]) > 0:
            US_Memorial_Day.append(self.day_of_week_month(weekmask="Mon", month=5, index=-1, Y=post_Old[0], Y_start=post_Old[1], Y_end=post_Old[2]))

        if len(US_Memorial_Day) > 0:
            return np.sort(np.concatenate(US_Memorial_Day))

        

    def US_Independence_Day(self, Y=None, Y_start=1776, Y_end=None, hard_end=None):

        return self.fixed_date(day=4, month=7, Y=Y, Y_start=Y_start, Y_end=Y_end, hard_start=True, hard_end=hard_end)


    def US_Independence_Day_OBS(self, Y=None):


        Old = self.Y_range(Y=Y, Y_start=1885, Y_end=1953, hard_start=True, hard_end=True)
        post_Old = self.Y_range(Y=Y, Y_start=1954, hard_start=True)

        US_Independence_Day_OBS = []
        if np.size(Old[0]) > 0:
            US_Independence_Day_Old = self.US_Independence_Day(Y=Old[0], Y_start=Old[1], Y_end=Old[2])
            US_Independence_Day_Old[np.is_busday(US_Independence_Day_Old, weekmask="Sun")] += np.timedelta64(1, "D")
            US_Independence_Day_OBS.append(US_Independence_Day_Old)
        if np.size(post_Old[0]) > 0:
            US_Independence_Day_post_Old = self.US_Independence_Day(Y=post_Old[0], Y_start=post_Old[1], Y_end=post_Old[2])
            self.OBS_closest_weekday_shift(US_Independence_Day_post_Old)
            US_Independence_Day_OBS.append(US_Independence_Day_post_Old)

        if len(US_Independence_Day_OBS) > 0:
            return np.sort(np.concatenate(US_Independence_Day_OBS))


    def US_Labor_Day(self, Y=None):

        return self.day_of_week_month(weekmask="Mon", month=9, index=0, Y=Y, Y_start=1971, hard_start=True)
    

    def US_Thanksgiving(self, Y=None):

        pre_FDR = self.Y_range(Y=Y, Y_start=1885, Y_end=1938, hard_start=True, hard_end=True)
        FDR = self.Y_range(Y=Y, Y_start=1939, Y_end=1941, hard_start=True, hard_end=True)
        post_FDR = self.Y_range(Y=Y, Y_start=1942, hard_start=True)

        US_Thanksgiving = []
        if np.size(pre_FDR[0]) > 0:
            US_Thanksgiving.append(self.day_of_week_month(weekmask="Thu", month=11, index=-1, Y=pre_FDR[0], Y_start=pre_FDR[1], Y_end=pre_FDR[2]))
        if np.size(FDR[0]) > 0:
            US_Thanksgiving.append(self.day_of_week_month(weekmask="Thu", month=11, index=-2, Y=FDR[0], Y_start=FDR[1], Y_end=FDR[2]))
        if np.size(post_FDR[0]) > 0:
            US_Thanksgiving.append(self.day_of_week_month(weekmask="Thu", month=11, index=3, Y=post_FDR[0], Y_start=post_FDR[1], Y_end=post_FDR[2]))

        if len(US_Thanksgiving) > 0:
            return np.sort(np.concatenate(US_Thanksgiving))
            


    def US_Christmas_Day_OBS(self, Y=None):

        Old = self.Y_range(Y=Y, Y_start=1885, Y_end=1953, hard_start=True, hard_end=True)
        post_Old = self.Y_range(Y=Y, Y_start=1954, hard_start=True)

        US_Christmas_Day_OBS = []
        if np.size(Old[0]) > 0:
            US_Christmas_Day_Old = self.Christmas_Day(Y=Old[0], Y_start=Old[1], Y_end=Old[2])
            US_Christmas_Day_Old[np.is_busday(US_Christmas_Day_Old, weekmask="Sun")] += np.timedelta64(1, "D")
            US_Christmas_Day_OBS.append(US_Christmas_Day_Old)
        if np.size(post_Old[0]) > 0:
            US_Christmas_Day_post_Old = self.Christmas_Day(Y=post_Old[0], Y_start=post_Old[1], Y_end=post_Old[2])
            self.OBS_closest_weekday_shift(US_Christmas_Day_post_Old)
            US_Christmas_Day_OBS.append(US_Christmas_Day_post_Old)

        if len(US_Christmas_Day_OBS) > 0:
            return np.sort(np.concatenate(US_Christmas_Day_OBS))



    
def get_calender_schedule(exchange, start_date=None, end_date=None):


    if start_date is None:
        start_date = "1900-01-01"
    if end_date is None:
        end_date = "2029-12-31"

    false_list = {i: {"positive": None, "negative": None} for i in mcal.get_calendar_names()}
    false_date = false_list.copy()

    calender = mcal.get_calendar(exchange)
    schedule = calender.schedule(start_date=start_date, end_date=end_date)

    # London Stock Exchange (LSE), England
    false_list["LSE"]["positive"] = [
                                        "1995-05-08", "1999-12-31", 
                                        "1977-06-07", "2002-06-03", "2002-06-04", 
                                        "1973-11-14", "1973-11-14","2011-04-29", 
                                        "1999-12-31"
                                    ]
    false_list["LSE"]["negative"] = [
                                        "1995-05-01", "2020-05-04", 
                                        "2002-05-27", "2012-05-28"
                                    ]

    for i in false_list:
        for j in false_list[i]:
            if false_list[i][j] is not None:
                false_date[i][j] = np.array([np.datetime64(k) for k in false_list[i][j]])
    
    schedule = schedule.iloc[np.logical_not(np.isin(schedule.index, false_date[exchange]["positive"]))]

    if false_date[exchange]["negative"] is not None:

        for k in false_date[exchange]["negative"]:
            if np.isin(k, schedule.index) == False:
                new_date = pd.Timestamp(k)

                past = schedule.index < new_date
                ref = np.sum(past)
                if np.all(past):
                    ref = np.size(past) - 1                    

                new_pd_data = schedule.iloc[ref] - pytz.utc.localize(schedule.index[ref])
                new_pd = pd.DataFrame([new_pd_data + new_date], index=[new_date], dtype="datetime64[ns, UTC]")
                schedule = schedule.append(new_pd)

        schedule = schedule.sort_index()
    
    return calender, schedule