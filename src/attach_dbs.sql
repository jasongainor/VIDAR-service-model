-- Attach the copied VIDA 2014D databases (read use only; files are copies).
-- DB names follow vida/db/dbutils/attachall.sql; servicerep_en-US added per
-- jboss sqlserver-ds.xml. Compat level bumped to 100 minimum on upgrade.
CREATE DATABASE [BaseData] ON
  (FILENAME = '/vida-db/BaseData_Data.MDF'),
  (FILENAME = '/vida-db/BaseData_log.LDF')
  FOR ATTACH;
GO
CREATE DATABASE [EPC] ON
  (FILENAME = '/vida-db/EPC_Data.mdf'),
  (FILENAME = '/vida-db/EPC_Log.ldf')
  FOR ATTACH;
GO
CREATE DATABASE [DiagSwdlRepository] ON
  (FILENAME = '/vida-db/DiagSwdlRepository_Data.MDF'),
  (FILENAME = '/vida-db/DiagSwdlRepository_log.LDF')
  FOR ATTACH;
GO
CREATE DATABASE [CarCom] ON
  (FILENAME = '/vida-db/CarComRT_Data.MDF'),
  (FILENAME = '/vida-db/CarComRT_Log.LDF')
  FOR ATTACH;
GO
CREATE DATABASE [servicerep_en-US] ON
  (FILENAME = '/vida-db/servicerep_en-US.MDF'),
  (FILENAME = '/vida-db/servicerep_en-US.LDF')
  FOR ATTACH;
GO
SELECT name, compatibility_level, state_desc FROM sys.databases WHERE database_id > 4;
GO
