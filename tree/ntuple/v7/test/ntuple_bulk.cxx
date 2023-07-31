#include "ntuple_test.hxx"

TEST(RNTupleBulk, Simple)
{
   FileRaii fileGuard("test_ntuple_bulk_simple.root");
   {
      auto model = RNTupleModel::Create();
      auto fldInt = model->MakeField<int>("int");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      for (int i = 0; i < 10; ++i) {
         *fldInt = i;
         writer->Fill();
      }
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   // TODO(jblomer): find a better way to expose the GenerateBulk method of the target field
   auto fieldZero = reader->GetModel()->GetFieldZero();
   std::unique_ptr<RFieldBase::RBulk> bulk;
   for (auto &f : *fieldZero) {
      if (f.GetName() != "int")
         continue;
      bulk = std::make_unique<RFieldBase::RBulk>(f.GenerateBulk());
   }

   auto mask = std::make_unique<bool[]>(10);
   std::fill(mask.get(), mask.get() + 10, true);
   auto intArr5 = static_cast<int *>(bulk->ReadBulk(RClusterIndex(0, 0), mask.get(), 5));
   for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, intArr5[i]);
   }

   auto intArr1 = static_cast<int *>(bulk->ReadBulk(RClusterIndex(0, 1), mask.get(), 1));
   EXPECT_EQ(1, intArr1[0]);
   EXPECT_EQ(static_cast<int *>(intArr5) + 1, static_cast<int *>(intArr1));

   auto intArr10 = static_cast<int *>(bulk->ReadBulk(RClusterIndex(0, 0), mask.get(), 10));
   for (int i = 0; i < 10; ++i) {
      EXPECT_EQ(i, intArr10[i]);
   }
}

TEST(RNTupleBulk, Complex)
{
   FileRaii fileGuard("test_ntuple_bulk_complex.root");
   {
      auto model = RNTupleModel::Create();
      auto fldS = model->MakeField<CustomStruct>("S");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      for (int i = 0; i < 10; ++i) {
         fldS->a = i;
         writer->Fill();
      }
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto fieldZero = reader->GetModel()->GetFieldZero();
   std::unique_ptr<RFieldBase::RBulk> bulk;
   for (auto &f : *fieldZero) {
      if (f.GetName() != "S")
         continue;
      bulk = std::make_unique<RFieldBase::RBulk>(f.GenerateBulk());
   }

   auto mask = std::make_unique<bool[]>(10);
   for (unsigned int i = 0; i < 10; ++i)
      mask[i] = (i % 2 == 0);

   auto SArr5 = static_cast<CustomStruct *>(bulk->ReadBulk(RClusterIndex(0, 0), mask.get(), 5));
   for (int i = 0; i < 5; ++i) {
      EXPECT_FLOAT_EQ((i % 2 == 0) ? float(i) : 0.0, SArr5[i].a);
   }

   auto SArr1 = static_cast<CustomStruct *>(bulk->ReadBulk(RClusterIndex(0, 1), mask.get() + 1, 1));
   EXPECT_FLOAT_EQ(0.0, SArr1[0].a);
   EXPECT_EQ(static_cast<CustomStruct *>(SArr5) + 1, static_cast<CustomStruct *>(SArr1));

   SArr1 = static_cast<CustomStruct *>(bulk->ReadBulk(RClusterIndex(0, 1), mask.get(), 1));
   EXPECT_FLOAT_EQ(1.0, SArr1[0].a);
   EXPECT_EQ(static_cast<CustomStruct *>(SArr5) + 1, static_cast<CustomStruct *>(SArr1));

   for (unsigned int i = 0; i < 10; ++i)
      mask[i] = !mask[i];
   auto SArr10 = static_cast<CustomStruct *>(bulk->ReadBulk(RClusterIndex(0, 0), mask.get(), 10));
   for (int i = 0; i < 10; ++i) {
      EXPECT_FLOAT_EQ((i % 2 == 0) ? 0.0 : float(i), SArr10[i].a);
   }
}

TEST(RNTupleBulk, CardinalityField)
{
   using RNTupleCardinality32 = ROOT::Experimental::RNTupleCardinality<std::uint32_t>;
   using RNTupleCardinality64 = ROOT::Experimental::RNTupleCardinality<std::uint64_t>;

   FileRaii fileGuard("test_ntuple_bulk_simple.root");
   {
      auto model = RNTupleModel::Create();
      auto fldVec = model->MakeField<ROOT::RVec<int>>("vint");
      model->AddProjectedField(std::make_unique<RField<RNTupleCardinality32>>("card32"),
                               [](const std::string &) { return "vint"; });
      model->AddProjectedField(std::make_unique<RField<RNTupleCardinality64>>("card64"),
                               [](const std::string &) { return "vint"; });
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      for (int i = 0; i < 10; ++i) {
         fldVec->resize(i);
         writer->Fill();
      }
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());

   auto fieldZero = reader->GetModel()->GetFieldZero();
   std::unique_ptr<RFieldBase::RBulk> bulk32;
   std::unique_ptr<RFieldBase::RBulk> bulk64;
   for (auto &f : *fieldZero) {
      if (f.GetName() == "card32")
         bulk32 = std::make_unique<RFieldBase::RBulk>(f.GenerateBulk());
      if (f.GetName() == "card64")
         bulk64 = std::make_unique<RFieldBase::RBulk>(f.GenerateBulk());
   }

   auto mask = std::make_unique<bool[]>(10);
   std::fill(mask.get(), mask.get() + 10, false /* the cardinality field optimization should ignore the mask */);

   auto card32Arr = static_cast<std::uint32_t *>(bulk32->ReadBulk(RClusterIndex(0, 0), mask.get(), 10));
   auto card64Arr = static_cast<std::uint64_t *>(bulk64->ReadBulk(RClusterIndex(0, 0), mask.get(), 10));
   for (int i = 0; i < 10; ++i) {
      EXPECT_EQ(i, card32Arr[i]);
      EXPECT_EQ(i, card64Arr[i]);
   }
}
